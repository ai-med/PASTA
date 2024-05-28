import math
from pathlib import Path
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast

from torchvision import transforms as T, utils

import torchio as tio
from einops import rearrange, reduce, repeat

from tqdm.auto import tqdm
from ema_pytorch import EMA

import nibabel as nib
import numpy as np

from accelerate import Accelerator

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

from torch.utils.tensorboard import SummaryWriter

from src.datasets.dataset import *
from src.utils.utils import *
from src.evals.ssim import *
from src.utils.data_utils import reconstruct_scan_from_2_5D_slices


class Trainer(object):
    def __init__(
        self,   
        diffusion_model,
        folder,
        *,
        input_slice_channel = 15,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = False,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.999),
        weight_decay = 1e-5,
        save_and_sample_every = 1000,
        num_samples = 16,
        results_folder = './results',
        amp = False,
        fp16 = True,
        split_batches = False,
        calculate_fid = False,
        inception_block_idx = 2048,
        dataset = None,
        image_direction = 'axial', # sagittal, axial, coronal
        num_slices = 1,
        model_cycling = False,
        tabular_cond = False,
        resume = None,
        pretrain = None,
        test_batch_size = 8,
        eval_mode = False,
        eval_dataset = 'ADNI',
        eval_resolution = [96, 112, 96],
        ROI_mask = None,
        dx_labels = ['CN', 'Dementia', 'MCI'],
    ):
        super().__init__()
        self.dataset = dataset
        self.fp16 = fp16   
        self.image_direction = image_direction
        self.input_slice_channel = input_slice_channel
        self.model_cycling = model_cycling
        self.tabular_cond = tabular_cond
        self.eval_mode = eval_mode
        self.eval_dataset = eval_dataset
        self.results_folder = results_folder
        self.ROI_mask = ROI_mask
        self.eval_resolution = eval_resolution

        if not eval_mode:
            self.writer = SummaryWriter(log_dir = results_folder + '/logs')

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )
        self.accelerator.native_amp = amp

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.model.out_channels

        # try to load pretrianed model
        if pretrain:
            ignored_weights = ['input_blocks.0.0.weight',
                                'out.2.weight',
                                'out.2.bias']
            self.load_pretrained_model(pretrain, ignored_weights)
            print('===load pretrained model successfully===')


        # InceptionV3 for fid-score computation
        self.inception_v3 = None
        if calculate_fid:
            assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
            self.inception_v3 = InceptionV3([block_idx])
            self.inception_v3.to(self.device)

        # sampling and training hyperparameters
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.model.image_size

        # dataset and dataloader
        if dataset == 'MRI2PET':
            if not eval_mode:
                train_data = folder + 'train.h5'
                valid_data = folder + 'valid.h5'
        
                self.ds = MRI2PET_2_5D_Dataset(self.image_size, data_path = train_data, output_dim=input_slice_channel,
                    direction = image_direction, num_slices = num_slices, 
                    random_flip = False, random_affine = False,
                    ROI_mask = ROI_mask, dx_labels = dx_labels)
                
                self.ds_valid = MRI2PET_2_5D_Dataset(self.image_size, data_path = valid_data, output_dim=input_slice_channel,
                    direction = image_direction, num_slices = num_slices, 
                    random_flip = False, random_affine = False,
                    dx_labels = dx_labels)
                
                
                self.dl_valid = DataLoader(self.ds_valid, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 2, drop_last=False)

            else:
                if self.eval_dataset == 'ADNI':
                    self.test_data = folder + 'test.h5' ###
                    # use whole scan per iteration
                    ds_test = SlicedScanMRI2PETDataset(eval_resolution, data_path = self.test_data, output_dim=input_slice_channel,
                        direction = image_direction, random_flip=None,
                        dx_labels = dx_labels)
                else:
                    raise NotImplementedError
                    
                self.dl_test = DataLoader(ds_test, batch_size = test_batch_size, shuffle = False, pin_memory = True, num_workers = 2, drop_last=False)
        
        else:
            raise NotImplementedError

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)
        
        
        if not eval_mode:
            dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 2, drop_last=False)
            dl = self.accelerator.prepare(dl)
            self.dl = dl


            # optimizer and scheduler
            self.opt = torch.optim.AdamW(diffusion_model.parameters(), lr = train_lr, betas = adam_betas, weight_decay = weight_decay)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=train_num_steps, eta_min=train_lr*0.1)

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

            # step counter state
            self.step = 0

            # prepare model, dataloader, optimizer with accelerator
            if resume:
                self.load(self.step, path = resume)
                print(f"resuming from model successfully")

            self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        
        print_model_size(self.model)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone, model_name = 'model.pt'):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }
    
        torch.save(data, str(self.results_folder / model_name))

    def load(self, milestone = 0, path = None):
        accelerator = self.accelerator
        device = accelerator.device

        if path is not None:
            data = torch.load(path, map_location = device)
        else:
            data = torch.load(str(self.results_folder / f'model.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        msg = model.load_state_dict(data['model'], strict=False)
        print('======load pretrained model successfully========')
        print("missing keys:", msg.missing_keys)
        print("Unexpected keys:", msg.unexpected_keys)
        

        self.step = data['step']

        msg_opt = self.opt.load_state_dict(data['opt'])

        if self.accelerator.is_main_process:
            msg_ema = self.ema.load_state_dict(data["ema"], strict=False)
            print("ema missing keys:", msg_ema.missing_keys)
            print("ema Unexpected keys:", msg_ema.unexpected_keys)

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
        
        print('loaded model successfully from step:', self.step)
    
    def load_pretrained_model(self, path, ignore):
        accelerator = self.accelerator
        device = accelerator.device

        pretrained_model = torch.load(path, map_location = device)
        new_state_dict = {}

        for key, value in pretrained_model.items():
            if key in ignore:
                continue
            new_key = 'model.' + key
            new_state_dict[new_key] = value

        model = self.accelerator.unwrap_model(self.model)
        msg = model.load_state_dict(new_state_dict, strict=False)
        print('======load pretrained model successfully========')
        print("missing keys:", msg.missing_keys)
        print("Unexpected keys:", msg.unexpected_keys)



    @torch.no_grad()
    def calculate_activation_statistics(self, samples):
        assert exists(self.inception_v3)

        self.inception_v3.eval()

        features = self.inception_v3(samples)[0]
        features = rearrange(features, '... 1 1 -> ...').cpu().numpy()

        mu = np.mean(features, axis = 0)
        sigma = np.cov(features, rowvar = False)
        return mu, sigma

    def fid_score(self, real_samples, fake_samples):

        if real_samples.shape[1] == 1:
            real_samples, fake_samples = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c = 3), (real_samples, fake_samples))

        min_batch = min(real_samples.shape[0], fake_samples.shape[0])
        real_samples, fake_samples = map(lambda t: t[:min_batch], (real_samples, fake_samples))

        m1, s1 = self.calculate_activation_statistics(real_samples)
        m2, s2 = self.calculate_activation_statistics(fake_samples)

        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value
    
    def psnr(self, img1, img2):
        """
        Args:
            img1: (n, c, h, w, l)
        """
        v_max = 1.
        # (n,)
        min_batch = min(img1.shape[0], img2.shape[0])
        img1, img2 = map(lambda t: t[:min_batch], (img1, img2))
        mse = torch.mean((img1 - img2)**2, dim=[1, 2, 3])
        return 20 * torch.log10(v_max / torch.sqrt(mse))

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        best_val_mae = 1000.
        best_val_ssim = 0.

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(iter(self.dl))
                    self.opt.zero_grad()

                    with autocast(self.fp16, dtype=torch.float16):
                    # with self.accelerator.autocast():
                        if self.dataset == 'MRI2PET' or self.dataset == 'GM2PET':
                            _train_data = data
                            mri_data, pet_data = _train_data[0], _train_data[1]
                            mri_uid = _train_data[-1]
                            cond_data = mri_data
                            data_label = _train_data[2]

                            if self.tabular_cond:
                                tabular_data = _train_data[3]
                            else: 
                                tabular_data = None
                            
                            slice_index = _train_data[4]
                            loss_weight_mask = _train_data[5]

                            model_output = self.model(pet_data, cond = cond_data, tab_cond = tabular_data,
                                                    loss_weight_mask = loss_weight_mask)
                            loss_dif = model_output['loss'].mean()
                            self.writer.add_scalar('train/train_loss_dif', loss_dif, self.step)
                            loss = loss_dif

                            
                            if self.model_cycling:
                                # first cycle, part two, from PET to MRI
                                synthetic_pet = model_output['model_output']
                                model_output_cycle = self.model(mri_data, cond = synthetic_pet, tab_cond = tabular_data, 
                                                            model_cycle = True,
                                                            loss_weight_mask = loss_weight_mask)
                                loss_cycle = model_output_cycle['loss'].mean()
                                self.writer.add_scalar('train/train_loss_cycle', loss_cycle, self.step)
                                loss += loss_cycle

                                # second cycle, part one, from PET to MRI
                                model_output_r2 = self.model(mri_data, cond = pet_data, tab_cond = tabular_data,
                                                            model_cycle = True,
                                                            loss_weight_mask = loss_weight_mask)
                                loss_cycle_r2 = model_output_r2['loss'].mean()
                                loss += loss_cycle_r2

                                # second cycle, part two, from MRI to PET
                                synthetic_mri = model_output_r2['model_output']
                                model_output_cycle_r2 = self.model(pet_data, cond = synthetic_mri, tab_cond = tabular_data,
                                                            model_cycle = False,
                                                            loss_weight_mask = loss_weight_mask)
                                loss += model_output_cycle_r2['loss'].mean()
                            
                            
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()
                        
                        
                        else:
                            data = next(self.dl).to(device)
                            loss = self.model(data)
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
                desc = f'loss: {loss_dif:.6f}'


                pbar.set_description(desc)
                self.writer.add_scalar('train/train_loss', total_loss, self.step)

                accelerator.wait_for_everyone()

                self.opt.step()
                self.scheduler.step()
                

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        all_images_list = []
                        data = []

                        if self.dataset == 'MRI2PET':
                            valid_loss = 0.
                            val_iter = 0
                            for valid_data in self.dl_valid:
                                _valid_data = valid_data
                                valid_data_mri, valid_data_pet = _valid_data[0], _valid_data[1]
                                valid_cond_data = valid_data_mri

                                mri_uid = _valid_data[-1]

                                if self.tabular_cond:
                                    tabular_data = _valid_data[3]
                                    tabular_data = tabular_data.to(device)
                                else: 
                                    tabular_data = None
                                
                                slice_index = _valid_data[4]

                                batches = valid_data_mri.shape[0]

                                with torch.no_grad():
                                    milestone = self.step // self.save_and_sample_every
                                    loss = self.ema.ema_model(valid_data_pet.to(device), cond = valid_cond_data.to(device), 
                                                        tab_cond = tabular_data)['loss'].mean()
                                    valid_loss = loss
                                    print(f'valid loss: {valid_loss:.6f}')

                                    # save validation loss to tensorboard
                                    self.writer.add_scalar('val/val_loss', valid_loss, self.step)

                                    print('valid_pet_shape:', valid_data_pet.shape)

                                    sample_pet = self.ema.ema_model.sample(shape = valid_data_pet.shape, cond = valid_cond_data.to(device), 
                                                        tab_cond = tabular_data)
                                    
                                                                   
                                    print('sample_pet_shape:', sample_pet.shape)
                                    

                                    all_images_list.append(sample_pet)
                                    data.append(valid_data_pet)


                                    if val_iter == 10:
                                        break
                                    
                                    val_iter += 1

                                
                                            
                        else:
                            with torch.no_grad():
                                milestone = self.step // self.save_and_sample_every
                                # batches = num_to_groups(self.num_samples, self.batch_size)
                                batches = [data.shape[0]]
                                all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        data = [img.reshape(img.shape[0]*img.shape[1], img.shape[2], img.shape[3]) for img in data]
                        all_images_list = [img.reshape(img.shape[0]*img.shape[1], img.shape[2], img.shape[3]) for img in all_images_list]                   
                        
                        
                        data = torch.cat(data, dim = 0).unsqueeze(1)                     
                        all_images = torch.cat(all_images_list, dim = 0).unsqueeze(1)

                        print('data_shape:', data.shape)
                        print('all_images_shape:', all_images.shape)


                        recon_loss_l2 = F.mse_loss(all_images, data.to(device))
                        print(f'valid recon L2 loss: {recon_loss_l2:.5f}')
                        recon_loss_l1 = F.l1_loss(all_images, data.to(device))
                        print(f'valid recon L1 loss: {recon_loss_l1:.5f}')


                        with open(str(self.results_folder / f'scores.txt'), 'a') as f:
                            f.write('\n')
                            f.write(f'milestone: {milestone}')                                   
                            f.write('\n')
                            f.write(f'valid recon L1 loss: {recon_loss_l1:.5f}')
                            f.write('\n')
                            f.write(f'valid recon L2 loss: {recon_loss_l2:.5f}')

                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                        if self.dataset == 'MRI2PET' or self.dataset == 'GM2PET':
                            utils.save_image(data, str(self.results_folder / f'real-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                        
                                       
                        self.save(milestone)

 
                        real_samples = torch.nan_to_num(data)
                        fake_samples = torch.nan_to_num(all_images)

                        if exists(self.inception_v3):
                            fid_score = self.fid_score(real_samples.to(device), fake_samples.to(device))
                            accelerator.print(f'fid_score: {fid_score}')
                            with open(str(self.results_folder / f'scores.txt'), 'a') as f:
                                f.write('\n')
                                f.write(f'fid_score: {fid_score}')
                        
                        psnr_score = self.psnr(data.to(device), all_images.to(device)).mean().detach().cpu().numpy()
                        accelerator.print(f'psnr_score: {str(psnr_score)}')
                        with open(str(self.results_folder / f'scores.txt'), 'a') as f:
                            f.write('\n')
                            f.write(f'psnr_score: {str(psnr_score)}')
                        
                        ssim_score = ssim(data.to(device), all_images.to(device)).detach().cpu().numpy()
                        accelerator.print(f'ssim_score: {str(ssim_score)}')
                        with open(str(self.results_folder / f'scores.txt'), 'a') as f:
                            f.write('\n')
                            f.write(f'ssim_score: {str(ssim_score)}')
                            f.write('\n')

                        mse_score = (data.to(device) - all_images.to(device)).pow(2).mean(dim=[1, 2, 3])
                        print(f'mse_score: {str(mse_score)}')
                        

                        if recon_loss_l1 - best_val_mae < 0.001 and ssim_score - best_val_ssim > 0.001:
                            best_val_mae = recon_loss_l1
                            best_val_ssim = ssim_score
                            print('best val mae:', best_val_mae.item())
                            self.save(milestone, model_name = 'best_val_model.pt')

                            with open(str(self.results_folder / f'scores.txt'), 'a') as f:
                                f.write('\n')
                                f.write(f'************Best val model, milestone: {milestone}*************')
                                f.write('\n')
                        
                pbar.update(1)

        accelerator.print('training complete')
    
    @torch.no_grad()
    def evaluate(self, checkpoint, evaluate_folder, synthesis=False, synthesis_folder=None, get_ROI_loss=False):
        self.evaluate_folder = Path(evaluate_folder)
        self.evaluate_folder.mkdir(exist_ok = True)

        if synthesis:
            assert synthesis_folder is not None, 'synthesis_folder is None'
            syn_folder = Path(synthesis_folder)
            syn_folder.mkdir(exist_ok = True)

        if get_ROI_loss:
            assert self.ROI_mask is not None, 'ROI_mask is None'
            self.ROI_mask = nib.load(self.ROI_mask).get_fdata()
            self.ROI_mask = tio.CropOrPad(self.eval_resolution)(self.ROI_mask[np.newaxis, ...]).squeeze(0)
            print('use ROI loss, ROI_mask_shape:', self.ROI_mask.shape)

            roi_losses = []
        
        data = torch.load(checkpoint, map_location = self.device)

        msg = self.model.load_state_dict(data['model'], strict=False)
        print('======load pretrained model successfully========')
        print("missing keys:", msg.missing_keys)
        print("Unexpected keys:", msg.unexpected_keys)
        
        self.step = data['step']
        
        msg_ema = self.ema.load_state_dict(data["ema"], strict=False)
        print("ema missing keys:", msg_ema.missing_keys)
        print("ema Unexpected keys:", msg_ema.unexpected_keys)
        
        print('loaded model successfully from:', checkpoint)

        self.model.eval()
        self.ema.ema_model.eval()
        
        if self.dataset == 'MRI2PET':
            test_loss = 0.
            flag = 0
            real_images_list = []
            sample_images_list = []
            error_map_list = []

            averaged_sample_images_list = []

            LABEL_MAP = {0: 'CN', 1: 'AD', 2: 'MCI'}


            for test_data in self.dl_test:
                _test_data = test_data
                test_data_mri, test_data_pet = _test_data[0], _test_data[1]

                dx_label = _test_data[2]
                mri_uid = _test_data[-1]


                if self.tabular_cond:
                    tabular_data = _test_data[3]
                    tabular_data = tabular_data.to(self.device)
                else:
                    tabular_data = None

                slices_per_sample_list = []
                GT_images_list = []

                if self.model_cycling:
                    slices_per_mri_sample_list = []
                    GT_mri_images_list = []


                for slice_num in range(len(test_data_mri)):
                    mri_slice = test_data_mri[slice_num]
                    pet_slice = test_data_pet[slice_num]

                    mri_slice = mri_slice.to(self.device)
                    pet_slice = pet_slice.to(self.device)

                    cond_slice = mri_slice

                    with torch.no_grad():                            
                        sample_pet = self.ema.ema_model.sample(shape = pet_slice.shape, cond = cond_slice, 
                                                                tab_cond = tabular_data)

                        if not synthesis:
                            test_loss = torch.abs(sample_pet - pet_slice)
                            print(f'****** Slice {slice_num}: *********')
                            print(f'=========== MRI -> PET, loss(GT PET, Syn PET): =============')
                            for batch in range(test_loss.shape[0]):
                                batch_loss = test_loss[batch].mean()
                                label = LABEL_MAP[int(dx_label[batch].item())]
                                print(f'{label}: {batch_loss:.6f} | ')
                        
                        # if model_cycling then calculate the PET -> MRI loss
                        if self.model_cycling and not synthesis:
                            model_output = self.ema.ema_model(cond_slice, cond = sample_pet, tab_cond = tabular_data, 
                                                    model_cycle = True)
                            cycle_loss = model_output['loss']
                            print(f'========== PET -> MRI, loss(GT MRI, Syn MRI): ==========')

                            for batch in range(cycle_loss.shape[0]):
                                batch_loss = cycle_loss[batch].mean()
                                label = LABEL_MAP[int(dx_label[batch].item())]
                                print(f'{label}: {batch_loss:.6f} | ')
                            
                            unnormalize = unnormalize_to_zero_to_one if self.ema.ema_model.rescale_intensity else identity
                            slices_per_mri_sample_list.append(unnormalize(model_output['model_output']))
                            

                        
                                        
                        slices_per_sample_list.append(sample_pet)                       
                        
                        if self.input_slice_channel > 1:
                            sample_pet = sample_pet[:, self.input_slice_channel // 2, ...].unsqueeze(1)
                            pet_slice = pet_slice[:, self.input_slice_channel // 2, ...].unsqueeze(1)

                            sample_images_list.append(sample_pet)
                            real_images_list.append(pet_slice)
                        
                            GT_images_list.append(pet_slice.squeeze(1))
                            
                            if self.model_cycling and not synthesis:
                                mri_slice = mri_slice[:, self.input_slice_channel // 2, ...].unsqueeze(1)
                                GT_mri_images_list.append(mri_slice.squeeze(1))
                        else:
                            sample_images_list.append(sample_pet)
                            real_images_list.append(pet_slice)

                            GT_images_list.append(pet_slice)
                            
                            if self.model_cycling and not synthesis:
                                GT_mri_images_list.append(mri_slice)
                    
                    if flag == 0 and not synthesis:
                        # error map become the difference between generated and real images
                        error_map = sample_pet.to(self.device) - pet_slice.to(self.device)
                        error_map_list.append(error_map)

                        error_map_label = _test_data[2]
                    

                if self.input_slice_channel > 1:
                    slices_per_sample_list = reconstruct_scan_from_2_5D_slices(slices_per_sample_list)
                   
                    for slice_per_sample in slices_per_sample_list:
                        averaged_sample_images_list.append(slice_per_sample.unsqueeze(1))
                    
                    if self.model_cycling and not synthesis:
                        slices_per_mri_sample_list = reconstruct_scan_from_2_5D_slices(slices_per_mri_sample_list)
                else:
                    averaged_sample_images_list = sample_images_list
                
                axis_map = {'coronal': -2, 'sagittal': -3, 'axial': -1}
                
                whole_pet_sample = np.stack([slice_tensor.cpu().numpy() for slice_tensor in slices_per_sample_list], axis=axis_map[self.image_direction])
                whole_GT_pet = np.stack([slice_tensor.cpu().numpy() for slice_tensor in GT_images_list], axis=axis_map[self.image_direction])

                assert whole_pet_sample.shape == whole_GT_pet.shape, (whole_pet_sample.shape, whole_GT_pet.shape)

                if self.model_cycling and not synthesis:
                    whole_mri_sample = np.stack([slice_tensor.cpu().numpy() for slice_tensor in slices_per_mri_sample_list], axis=axis_map[self.image_direction])
                    whole_GT_mri = np.stack([slice_tensor.cpu().numpy() for slice_tensor in GT_mri_images_list], axis=axis_map[self.image_direction])
                    assert whole_mri_sample.shape == whole_GT_mri.shape, (whole_mri_sample.shape, whole_GT_mri.shape)
                
                affine = np.array([
                            [1.5, 0, 0, 0],
                            [0, 1.5, 0, 0],
                            [0, 0, 1.5, 0],
                            [0, 0, 0, 1]
                        ])
                
                for slice_batch in range(whole_pet_sample.shape[0]):
                    
                    img_label = str(LABEL_MAP[int(dx_label[slice_batch].detach().cpu().numpy())])
                    _mri_uid = str(mri_uid[slice_batch])

                    if synthesis:
                        _L1_loss = np.abs(whole_pet_sample[slice_batch] - whole_GT_pet[slice_batch])
                        L1_loss = _L1_loss.mean()
                        pet_sample_file_name = str(syn_folder / f'{_mri_uid}_{img_label}_syn_pet.nii.gz')
                        pet_sample_img = nib.Nifti1Image(whole_pet_sample[slice_batch].squeeze(), affine=affine)
                        pet_sample_img.to_filename(pet_sample_file_name)
                        print(f'MRI -> PET: {_mri_uid}_{img_label}: {L1_loss:.6f}')
                        with open(str(syn_folder / f'all_eval_scores.txt'), 'a') as f:
                            f.write('\n')
                            f.write(f'MRI -> PET: {_mri_uid}_{img_label}: {L1_loss:.6f}')
                        if get_ROI_loss:
                            ROI_loss = _L1_loss * self.ROI_mask
                            ROI_loss = ROI_loss.mean()
                            print(f'MRI -> PET ROI loss: {_mri_uid}_{img_label}: {ROI_loss:.6f}')
                            with open(str(syn_folder / f'all_eval_scores.txt'), 'a') as f:
                                f.write('\n')
                                f.write(f'MRI -> PET ROI loss: {_mri_uid}_{img_label}: {ROI_loss:.6f}')
                            roi_losses.append(ROI_loss)
                    else:

                        whole_pet_sample_img = nib.Nifti1Image(whole_pet_sample[slice_batch].squeeze(), affine=affine)
                        whole_pet_sample_img.to_filename(str(self.evaluate_folder / f'whole_sample_pet_{_mri_uid}_{slice_batch}_{img_label}.nii.gz'))

                        whole_GT_pet_img = nib.Nifti1Image(whole_GT_pet[slice_batch].squeeze(), affine=affine)
                        whole_GT_pet_img.to_filename(str(self.evaluate_folder / f'whole_GT_pet_{_mri_uid}_{slice_batch}_{img_label}.nii.gz'))

                        error_map_img = nib.Nifti1Image((whole_pet_sample[slice_batch] - whole_GT_pet[slice_batch]).squeeze(), affine=affine)                     
                        error_map_img.to_filename(str(self.evaluate_folder / f'error_map_{_mri_uid}_{slice_batch}_{img_label}.nii.gz'))

                        _L1_loss = np.abs(whole_pet_sample[slice_batch] - whole_GT_pet[slice_batch])
                        L1_loss = _L1_loss.mean()
                        print(f'MRI -> PET: {_mri_uid}_{img_label}: {L1_loss:.6f}')

                        with open(str(self.evaluate_folder / f'eval_scores.txt'), 'a') as f:
                            f.write('\n')
                            f.write(f'MRI -> PET: {_mri_uid}_{img_label}: {L1_loss:.6f}')
                        
                        if get_ROI_loss:
                            ROI_loss = _L1_loss * self.ROI_mask
                            ROI_loss = ROI_loss.mean()
                            print(f'MRI -> PET ROI loss: {_mri_uid}_{img_label}: {ROI_loss:.6f}')
                            with open(str(self.evaluate_folder / f'eval_scores.txt'), 'a') as f:
                                f.write('\n')
                                f.write(f'MRI -> PET ROI loss: {_mri_uid}_{img_label}: {ROI_loss:.6f}')
                            roi_losses.append(ROI_loss)
                        


                    if self.model_cycling and not synthesis:
                        L1_loss_mri = np.abs(whole_mri_sample[slice_batch] - whole_GT_mri[slice_batch]).mean()
                        print(f'PET -> MRI: {_mri_uid}_{img_label}: {L1_loss_mri:.6f}')
                        print('====================')
                        with open(str(self.evaluate_folder / f'eval_scores.txt'), 'a') as f:
                            f.write('\n')
                            f.write(f'PET -> MRI: {_mri_uid}_{img_label}: {L1_loss_mri:.6f}')
                            f.write('\n')
                            f.write('====================')


                if not synthesis:
                    if flag == 0:
                        break
                                 
                flag += 1

            sample_images_list = torch.cat(sample_images_list, dim = 0).to(self.device)
            real_images_list = torch.cat(real_images_list, dim = 0).to(self.device)

            averaged_sample_images_list = torch.cat(averaged_sample_images_list, dim = 0).to(self.device)

            if not synthesis:
                try:
                    import matplotlib.pyplot as plt
                    e_map = error_map_list
                    for num in range(len(e_map)):
                        for slice_batch in range(e_map[num].shape[0]):                       
                            save_dir = os.path.join(self.evaluate_folder, f'error_map_{slice_batch}')
                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir)
                            save_path = os.path.join(save_dir, f'slice_{num}.png')

                            plt.imshow(e_map[num][slice_batch].squeeze().detach().cpu().numpy(), cmap='seismic', vmin=-1, vmax=1)
                            # plt.colorbar()
                            LABEL_MAP = {0: 'CN', 1: 'AD', 2: 'MCI'}
                            plt.text(0, 0, str(LABEL_MAP[int(error_map_label[slice_batch].detach().cpu().numpy())]), color='red', fontsize=15, va='top')
                            plt.savefig(save_path)
                            plt.close()
                except ImportError:
                    print('matplotlib not installed, skip saving error heatmap')
                
                error_map_list = torch.stack(error_map_list, dim = 0).to(self.device)
                print(error_map_list.shape)
                for batch in range(error_map_list.shape[1]):
                    utils.save_image(error_map_list[:, batch, ...], os.path.join(self.evaluate_folder, 'error_map.png'), nrow = int(math.sqrt(self.num_samples)))

            if not synthesis:
                recon_loss_l2 = F.mse_loss(sample_images_list, real_images_list)
                print(f'test recon L2 loss: {recon_loss_l2:.4f}')
                recon_loss_l1 = F.l1_loss(sample_images_list, real_images_list)
                print(f'test recon L1 loss: {recon_loss_l1:.4f}')

                with open(str(self.evaluate_folder / f'eval_scores.txt'), 'a') as f:
                    f.write('\n')
                    f.write(f'test recon L2 loss: {recon_loss_l2:.4f}')
                    f.write('\n')
                    f.write(f'test recon L1 loss: {recon_loss_l1:.4f}')
                
                psnr_score = self.psnr(real_images_list, sample_images_list).mean().detach().cpu().numpy()
                print(f'psnr_score: {str(psnr_score)}')
                with open(str(self.evaluate_folder / f'eval_scores.txt'), 'a') as f:
                    f.write('\n')
                    f.write(f'psnr_score: {str(psnr_score)}')
                
                ssim_score = ssim(real_images_list, sample_images_list).detach().cpu().numpy()
                print(f'ssim_score: {str(ssim_score)}')
                with open(str(self.evaluate_folder / f'eval_scores.txt'), 'a') as f:
                    f.write('\n')
                    f.write(f'ssim_score: {str(ssim_score)}')
                    f.write('\n')

            ### evaluate the averaged sample images

            recon_loss_l2 = F.mse_loss(averaged_sample_images_list, real_images_list)
            print(f'test recon L2 loss (averaged sample): {recon_loss_l2:.4f}')
            recon_loss_l1 = F.l1_loss(averaged_sample_images_list, real_images_list)
            print(f'test recon L1 loss (averaged sample): {recon_loss_l1:.4f}')

            with open(str(self.evaluate_folder / f'eval_scores.txt'), 'a') as f:
                f.write('\n')
                f.write(f'test recon L2 loss (averaged sample): {recon_loss_l2:.4f}')
                f.write('\n')
                f.write(f'test recon L1 loss (averaged sample): {recon_loss_l1:.4f}')
            
            if get_ROI_loss:
                roi_losses = sum(roi_losses) / len(roi_losses)
                print(f'test ROI loss: {roi_losses: .4f}')
                with open(str(self.evaluate_folder / f'eval_scores.txt'), 'a') as f:
                    f.write('\n')
                    f.write(f'test ROI loss: {roi_losses: .6f}')
                    f.write('\n')
            
            
            psnr_score = self.psnr(real_images_list, averaged_sample_images_list).mean().detach().cpu().numpy()
            print(f'psnr_score (averaged sample): {str(psnr_score)}')
            with open(str(self.evaluate_folder / f'eval_scores.txt'), 'a') as f:
                f.write('\n')
                f.write(f'psnr_score (averaged sample): {str(psnr_score)}')
            
            ssim_score = ssim(real_images_list, averaged_sample_images_list).detach().cpu().numpy()
            print(f'ssim_score (averaged sample): {str(ssim_score)}')
            with open(str(self.evaluate_folder / f'eval_scores.txt'), 'a') as f:
                f.write('\n')
                f.write(f'ssim_score (averaged sample): {str(ssim_score)}')
                f.write('\n')





        