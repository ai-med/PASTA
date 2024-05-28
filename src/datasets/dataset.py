import h5py
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from torchvision import transforms as T, utils
from tqdm import tqdm

import pandas as pd

from src.utils.data_utils import get_neighboring_slices, multislice_data_minimal_process, \
                                                    get_3d_image_transform, crop_minimally_keep_ratio, rescale_intensity, rescale_intensity_3D, \
                                                    resample_and_reshape, process_tabular_data, data_minimal_process, CropOrPad_3D

import nibabel as nib
import numpy as np
import pandas as pd
import os

import logging

LOG = logging.getLogger(__name__)

# constants

DIAGNOSIS_MAP = {"CN": 0, "Dementia": 1, "AD": 1, "MCI": 2}
DIAGNOSIS_MAP_binary = {"CN": 0, "AD": 1, "Dementia": 1}

# dataset

class SlicedScanMRI2PETDataset(Dataset):
    def __init__(
        self,
        resolution = None,
        mri_root_path = '/path/to/mri',
        pet_root_path = '/path/to/pet',
        data_path = None,
        output_dim = 1,
        direction = 'coronal',
        standardized_tabular = True,
        classes=None, # 'binary' or 'multi' (with MCI)
        random_crop=False,
        random_flip=False,
        random_affine=False,
        resample_mri=False,
        dx_labels = ['CN', 'MCI', 'Dementia'],
    ):
        super().__init__()
        self.resolution = resolution
        self.mri_root_path = mri_root_path
        self.pet_root_path = pet_root_path
        self.data_path = data_path
        self.output_dim = output_dim
        self.direction = direction
        self.standardized_tabular = standardized_tabular
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.random_affine = random_affine
        self.with_label = classes
        self.resample_mri = resample_mri
        self.dx_labels = dx_labels

        self._load()

    def _load(self):
        mri_data = []
        pet_data = []
        diagnosis = []
        tabular_data = []
        mri_uid = []

        PET_shape = (113, 137, 113)
        MRI_shape = (113, 137, 113)

        if self.data_path is not None and 'h5' in self.data_path:       
            print('loaded from h5 file')
            with h5py.File(self.data_path, mode='r') as file:
                for name, group in tqdm(file.items(), total=len(file)):
                    if name == "stats":
                        self.tabular_mean = group["tabular/mean"][:]
                        self.tabular_std = group["tabular/stddev"][:]
                    else:
                        if group.attrs['DX'] not in self.dx_labels:
                            continue
                        if self.resample_mri:
                            _raw_mri_data = group['MRI/T1/data'][:]
                            _resampled_mri_data = resample_and_reshape(_raw_mri_data, (1.5, 1.5, 1.5), PET_shape)
                            input_mri_data = _resampled_mri_data
                            MRI_shape = PET_shape
                            assert input_mri_data.shape == PET_shape
                        else:
                            input_mri_data = group['MRI/T1/data'][:]
                        
                        _pet_data = group['PET/FDG/data'][:]
                        _mri_data = input_mri_data
                        _tabular_data = group['tabular'][:]
                        _diagnosis = group.attrs['DX']

                        _pet_data = np.nan_to_num(_pet_data, copy=False)
                        mri_data.append(_mri_data)
                        pet_data.append(_pet_data)
                        tabular_data.append(_tabular_data)
                        diagnosis.append(_diagnosis)
                        mri_uid.append(name)
        else:
            print('loaded from: ', self.mri_root_path, self.pet_root_path)

            mri_id = os.listdir(self.mri_root_path)
            mri_input = [os.path.join(self.mri_root_path, i, 'mri.nii.gz') for i in mri_id]
            pet_input = [os.path.join(self.pet_root_path, i[:-8], f'pet_fdg.nii.gz') for i in mri_id]
            mri_data = [nib.load(i).get_fdata() for i in mri_input]
            pet_data = [nib.load(i).get_fdata() for i in pet_input]

            csv_info = pd.read_csv('data_info.csv')
            diagnosis = [csv_info.loc[csv_info["IMAGEUID"] == i]['DX'].values[0] for i in mri_id]
            tabular_data = [csv_info.loc[csv_info["IMAGEUID"] == i]['TAB'].values[0] for i in mri_id]
            mri_uid = mri_id



        self.len_data = len(pet_data)
        self._image_data_mri = mri_data
        self._image_data_pet = pet_data
        self._tabular_data = tabular_data
        self._mri_uid = mri_uid
        
      
        LOG.info("DATASET: %s", self.data_path if self.data_path is not None else self.mri_root_path)
        LOG.info("SAMPLES: %d", self.len_data)

        # if self.with_label is not None:
        labels, counts = np.unique(diagnosis, return_counts=True)
        LOG.info("Classes: %s", pd.Series(counts, index=labels))     

        if self.with_label == 'binary':
            self._diagnosis = [DIAGNOSIS_MAP_binary[d] for d in diagnosis]
        elif self.with_label == 'multi':
            self._diagnosis = [DIAGNOSIS_MAP[d] for d in diagnosis]
        else:
            self._diagnosis = [DIAGNOSIS_MAP[d] for d in diagnosis]

    
    def __len__(self):
        return self.len_data


    def __getitem__(self, idx):

        MRI_shape = self.resolution
        PET_shape = self.resolution

        mri_scan = self._image_data_mri[idx]
        pet_scan = self._image_data_pet[idx]
        tabular_data = self._tabular_data[idx]
        mri_uid = self._mri_uid[idx]

        mri_scan = rescale_intensity_3D(mri_scan)
        pet_scan = rescale_intensity_3D(pet_scan)
        mri_scan = CropOrPad_3D(mri_scan, MRI_shape)
        pet_scan = CropOrPad_3D(pet_scan, PET_shape)

        if self.standardized_tabular:
            tabular_data = (tabular_data - self.tabular_mean) / self.tabular_std
            tabular_data = process_tabular_data(tabular_data)

        mri_scan_list = []
        pet_scan_list = []

        data_transform = T.Compose([
                T.ToTensor(),
                # T.CenterCrop((self.resolution[0], self.resolution[1])) if self.resolution is not None else nn.Identity(),
                # T.Resize((self.resolution[0], self.resolution[1]), antialias=True) if self.resolution is not None else nn.Identity(),
                T.RandomVerticalFlip() if self.random_flip else nn.Identity(),
                T.RandomAffine(180, translate=(0.3, 0.3)) if self.random_affine else nn.Identity(),         
            ])

        if self.direction == 'coronal':
            for i in range(MRI_shape[1]):
                if self.output_dim > 1:
                    _mri_data = get_neighboring_slices(self.output_dim, self.direction, i, mri_scan)
                    _pet_data = get_neighboring_slices(self.output_dim, self.direction, i, pet_scan)
                    _pet_data = np.nan_to_num(_pet_data, copy=False)
                    _mri_data = multislice_data_minimal_process(self.output_dim, self.resolution, _mri_data, data_transform)
                    _pet_data = multislice_data_minimal_process(self.output_dim, self.resolution, _pet_data, data_transform)

                else:
                    _pet_data = pet_scan[:, i, :]
                    _mri_data = mri_scan[:, i, :]
                    _pet_data = np.nan_to_num(_pet_data, copy=False)            
                    _mri_data = data_minimal_process(self.resolution, _mri_data, data_transform)
                    _pet_data = data_minimal_process(self.resolution, _pet_data, data_transform)

                mri_scan_list.append(_mri_data)
                pet_scan_list.append(_pet_data)
        
        elif self.direction == 'sagittal':
            for i in range(MRI_shape[0]):
                if self.output_dim > 1:
                    _mri_data = get_neighboring_slices(self.output_dim, self.direction, i, mri_scan)
                    _pet_data = get_neighboring_slices(self.output_dim, self.direction, i, pet_scan)
                    _pet_data = np.nan_to_num(_pet_data, copy=False)
                    _mri_data = multislice_data_minimal_process(self.output_dim, self.resolution, _mri_data, data_transform)
                    _pet_data = multislice_data_minimal_process(self.output_dim, self.resolution, _pet_data, data_transform)
                else:
                    _pet_data = pet_scan[i, :, :]
                    _mri_data = mri_scan[i, :, :]
                    _pet_data = np.nan_to_num(_pet_data, copy=False)
                    _mri_data = data_minimal_process(self.resolution, _mri_data, data_transform)
                    _pet_data = data_minimal_process(self.resolution, _pet_data, data_transform)

                mri_scan_list.append(_mri_data)
                pet_scan_list.append(_pet_data)

        
        elif self.direction == 'axial':
            for i in range(MRI_shape[2]):
                if self.output_dim > 1:
                    _mri_data = get_neighboring_slices(self.output_dim, self.direction, i, mri_scan)
                    _pet_data = get_neighboring_slices(self.output_dim, self.direction, i, pet_scan)
                    
                    _pet_data = np.nan_to_num(_pet_data, copy=False)
                    _mri_data = multislice_data_minimal_process(self.output_dim, self.resolution, _mri_data, data_transform)
                    _pet_data = multislice_data_minimal_process(self.output_dim, self.resolution, _pet_data, data_transform)
                else:
                    _pet_data = pet_scan[:, :, i]
                    _mri_data = mri_scan[:, :, i]

                    _pet_data = np.nan_to_num(_pet_data, copy=False)
                    _mri_data = data_minimal_process(self.resolution, _mri_data, data_transform)
                    _pet_data = data_minimal_process(self.resolution, _pet_data, data_transform)

                mri_scan_list.append(_mri_data)
                pet_scan_list.append(_pet_data)

        label = self._diagnosis[idx]
       
        return mri_scan_list, pet_scan_list, label, tabular_data, mri_uid


class MRI2PET_2_5D_Dataset(Dataset):
    def __init__(
        self,
        resolution = None,
        mri_root_path = '/path/to/mri',
        pet_root_path = '/path/to/pet',
        data_path = None,
        output_dim = 3,
        direction = 'axial',
        num_slices = 'all',
        standardized_tabular = True,
        classes=None, # 'binary' or 'multi' (with MCI)
        random_crop=False,
        random_flip=False,
        random_affine=False,
        resample_mri=False,
        ROI_mask = None,
        dx_labels = ['CN', 'Dementia', 'MCI'],
    ):
        super().__init__()
        self.resolution = resolution
        self.mri_root_path = mri_root_path
        self.pet_root_path = pet_root_path
        self.data_path = data_path
        self.output_dim = output_dim
        self.direction = direction
        self.num_slices = num_slices
        self.standardized_tabular = standardized_tabular
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.random_affine = random_affine
        self.with_label = classes
        self.resample_mri = resample_mri

        self.ROI_mask = ROI_mask
        self.dx_labels = dx_labels

        self._load()

    def _load(self):
        mri_data = []
        pet_data = []
        diagnosis = []
        tabular_data = []
        slice_index = []
        mri_uid = []

        PET_shape = (113, 137, 113)
        MRI_shape = (113, 137, 113)

        flag = 0

        if 'h5' in self.data_path:       
            print('loaded from h5 file')

            with h5py.File(self.data_path, mode='r') as file:
                for name, group in tqdm(file.items(), total=len(file)):

                    if name == "stats":
                        self.tabular_mean = group["tabular/mean"][:]
                        self.tabular_std = group["tabular/stddev"][:]
                    else:
                        if group.attrs['DX'] not in self.dx_labels:
                            continue
                        if self.resample_mri:
                            _raw_mri_data = group['MRI/T1/data'][:]
                            _resampled_mri_data = resample_and_reshape(_raw_mri_data, (1.5, 1.5, 1.5), PET_shape)
                            input_mri_data = _resampled_mri_data
                            MRI_shape = PET_shape
                            assert input_mri_data.shape == PET_shape
                            input_pet_data = group['PET/FDG/data'][:]

                        else:
                            input_mri_data = group['MRI/T1/data'][:]
                            input_pet_data = group['PET/FDG/data'][:]


                        if self.direction == 'coronal':
                            max_slice_index = PET_shape[1] - 1
                            if self.num_slices == 1:
                                _mri_data = get_neighboring_slices(self.output_dim, self.direction, PET_shape[1] // 2 + 1, input_mri_data)
                                _pet_data = get_neighboring_slices(self.output_dim, self.direction, PET_shape[1] // 2 + 1, input_pet_data)
                                _tabular_data = group['tabular'][:]
                                _diagnosis = group.attrs['DX']

                                _pet_data = np.nan_to_num(_pet_data, copy=False)
                                mri_data.append(_mri_data)
                                pet_data.append(_pet_data)
                                tabular_data.append(_tabular_data)
                                diagnosis.append(_diagnosis)
                                mri_uid.append(name)
                                slice_index.append(PET_shape[1] // 2 + 1)

                            elif self.num_slices == 'all':
                                for i in range(PET_shape[1]):
                                    # get the ith slice's neighboring slices to form the image with self.output_dim channels
                                    _mri_data = get_neighboring_slices(self.output_dim, self.direction, i, input_mri_data)
                                    _pet_data = get_neighboring_slices(self.output_dim, self.direction, i, input_pet_data)
                                    
                                    _tabular_data = group['tabular'][:]
                                    _diagnosis = group.attrs['DX']

                                    _pet_data = np.nan_to_num(_pet_data, copy=False)
                                    if not np.any(_pet_data) or not np.any(_mri_data):
                                        continue
                                    mri_data.append(_mri_data)
                                    pet_data.append(_pet_data)
                                    tabular_data.append(_tabular_data)
                                    diagnosis.append(_diagnosis)
                                    mri_uid.append(name)
                                    slice_index.append(i)
                            else:
                                for i in range(-self.num_slices // 2, self.num_slices // 2):
                                    _mri_data = get_neighboring_slices(self.output_dim, self.direction, PET_shape[1] // 2 + 1 + i, input_mri_data)
                                    _pet_data = get_neighboring_slices(self.output_dim, self.direction, PET_shape[1] // 2 + 1 + i, input_pet_data)
                                    _tabular_data = group['tabular'][:]
                                    _diagnosis = group.attrs['DX']

                                    _pet_data = np.nan_to_num(_pet_data, copy=False)
                                    if not np.any(_pet_data) or not np.any(_mri_data):
                                        continue

                                    mri_data.append(_mri_data)
                                    pet_data.append(_pet_data)
                                    tabular_data.append(_tabular_data)
                                    diagnosis.append(_diagnosis)
                                    mri_uid.append(name)
                                    slice_index.append(PET_shape[1] // 2 + 1 + i)
                        
                        elif self.direction == 'sagittal':
                            max_slice_index = PET_shape[0] - 1
                            if self.num_slices == 1:
                                _mri_data = get_neighboring_slices(self.output_dim, self.direction, PET_shape[0] // 2 + 1, input_mri_data)
                                _pet_data = get_neighboring_slices(self.output_dim, self.direction, PET_shape[0] // 2 + 1, input_pet_data)
                                _tabular_data = group['tabular'][:]
                                _diagnosis = group.attrs['DX']

                                _pet_data = np.nan_to_num(_pet_data, copy=False)
                                mri_data.append(_mri_data)
                                pet_data.append(_pet_data)
                                tabular_data.append(_tabular_data)
                                diagnosis.append(_diagnosis)
                                mri_uid.append(name)
                                slice_index.append(PET_shape[0] // 2 + 1)
                                
                            elif self.num_slices == 'all':
                                for i in range(PET_shape[0]):
                                    _mri_data = get_neighboring_slices(self.output_dim, self.direction, i, input_mri_data)
                                    _pet_data = get_neighboring_slices(self.output_dim, self.direction, i, input_pet_data)
                                    _tabular_data = group['tabular'][:]
                                    _diagnosis = group.attrs['DX']

                                    _pet_data = np.nan_to_num(_pet_data, copy=False)
                                    if not np.any(_pet_data) or not np.any(_mri_data):
                                        continue
                                    mri_data.append(_mri_data)
                                    pet_data.append(_pet_data)
                                    tabular_data.append(_tabular_data)
                                    diagnosis.append(_diagnosis)
                                    mri_uid.append(name)
                                    slice_index.append(i)
                            else:
                                for i in range(-self.num_slices // 2, self.num_slices // 2):
                                    _mri_data = get_neighboring_slices(self.output_dim, self.direction, PET_shape[0] // 2 + 1 + i, input_mri_data)
                                    _pet_data = get_neighboring_slices(self.output_dim, self.direction, PET_shape[0] // 2 + 1 + i, input_pet_data)
                                    _tabular_data = group['tabular'][:]
                                    _diagnosis = group.attrs['DX']

                                    _pet_data = np.nan_to_num(_pet_data, copy=False)
                                    if not np.any(_pet_data) or not np.any(_mri_data):
                                        continue
                                    mri_data.append(_mri_data)
                                    pet_data.append(_pet_data)
                                    tabular_data.append(_tabular_data)
                                    diagnosis.append(_diagnosis)
                                    mri_uid.append(name)
                                    slice_index.append(PET_shape[0] // 2 + 1 + i)
                        
                        elif self.direction == 'axial':
                            max_slice_index = PET_shape[2] - 1
                            if self.num_slices == 1:
                                _mri_data = get_neighboring_slices(self.output_dim, self.direction, PET_shape[2] // 2 + 1, input_mri_data)
                                _pet_data = get_neighboring_slices(self.output_dim, self.direction, PET_shape[2] // 2 + 1, input_pet_data) 
                                _tabular_data = group['tabular'][:]
                                _diagnosis = group.attrs['DX']

                                _pet_data = np.nan_to_num(_pet_data, copy=False)
                                mri_data.append(_mri_data)
                                pet_data.append(_pet_data)
                                tabular_data.append(_tabular_data)
                                diagnosis.append(_diagnosis)
                                mri_uid.append(name)
                                slice_index.append(PET_shape[2] // 2 + 1)

                            elif self.num_slices == 'all':
                                for i in range(PET_shape[2]):
                                    _mri_data = get_neighboring_slices(self.output_dim, self.direction, i, input_mri_data)
                                    _pet_data = get_neighboring_slices(self.output_dim, self.direction, i, input_pet_data)
                                    _tabular_data = group['tabular'][:]
                                    _diagnosis = group.attrs['DX']

                                    _pet_data = np.nan_to_num(_pet_data, copy=False)
                                    if not np.any(_pet_data) or not np.any(_mri_data):
                                        continue

                                    mri_data.append(_mri_data)
                                    pet_data.append(_pet_data)
                                    
                                    tabular_data.append(_tabular_data)
                                    diagnosis.append(_diagnosis)
                                    mri_uid.append(name)
                                    slice_index.append(i)

                            else:
                                for i in range(-self.num_slices // 2, self.num_slices // 2):
                                    _mri_data = get_neighboring_slices(self.output_dim, self.direction, PET_shape[2] // 2 + 1 + i, input_mri_data)
                                    _pet_data = get_neighboring_slices(self.output_dim, self.direction, PET_shape[2] // 2 + 1 + i, input_pet_data)
                                    _tabular_data = group['tabular'][:]
                                    _diagnosis = group.attrs['DX']

                                    _pet_data = np.nan_to_num(_pet_data, copy=False)
                                    if not np.any(_pet_data) or not np.any(_mri_data):
                                        continue

                                    mri_data.append(_mri_data)
                                    pet_data.append(_pet_data)
                                    tabular_data.append(_tabular_data)
                                    diagnosis.append(_diagnosis)
                                    mri_uid.append(name)
                                    slice_index.append(PET_shape[2] // 2 + 1 + i)
                                 
                        

        else:
            raise NotImplementedError

        self.len_data = len(pet_data)
        self._image_data_mri = mri_data
        self._image_data_pet = pet_data
        self._tabular_data = tabular_data
        
        self._slice_index = [float(i / max_slice_index) for i in slice_index]
        self._max_slice_index = max_slice_index
        self._mri_uid = mri_uid
      
        LOG.info("DATASET: %s", self.data_path)
        LOG.info("SAMPLES: %d", self.len_data)

        LOG.info("Input Shape: {}".format(mri_data[0].shape))

        # if self.with_label is not None:
        labels, counts = np.unique(diagnosis, return_counts=True)
        LOG.info("Classes: %s", pd.Series(counts, index=labels))     

        if self.with_label == 'binary':
            self._diagnosis = [DIAGNOSIS_MAP_binary[d] for d in diagnosis]
        elif self.with_label == 'multi':
            self._diagnosis = [DIAGNOSIS_MAP[d] for d in diagnosis]
        else:
            self._diagnosis = [DIAGNOSIS_MAP[d] for d in diagnosis]

        if self.ROI_mask is not None:
            self._ROI_mask = nib.load(self.ROI_mask).get_fdata()
            print('Loaded ROI mask shape: ', self._ROI_mask.shape)
            assert PET_shape == self._ROI_mask.shape, ('ROI mask shape is not Input data shape', self._ROI_mask.shape, mri_data[0].shape)
        else:
            self._ROI_mask = None
    
    def __len__(self):
        return self.len_data


    def __getitem__(self, idx):

        mri_scan = self._image_data_mri[idx]
        pet_scan = self._image_data_pet[idx]
        tabular_data = self._tabular_data[idx]

        slice_index = self._slice_index[idx]
        assert slice_index <= 1 and slice_index >= 0, 'slice index should be normalized to [0, 1]'

        if self._ROI_mask is not None:
            roi_mask = get_neighboring_slices(self.output_dim, self.direction, int(slice_index * self._max_slice_index), self._ROI_mask)
        
            assert roi_mask.shape == mri_scan.shape, ('roi mask shape is not Input scan shape', roi_mask.shape, mri_scan.shape)

            loss_weight_mask = roi_mask.copy()
            loss_weight_mask[roi_mask == 0] = 1
            loss_weight_mask[roi_mask == 1] = 10
        else:
            loss_weight_mask = np.ones(mri_scan.shape)
            print('No ROI mask is used, loss weight mask is all ones')
        

        if self.standardized_tabular:
            tabular_data = (tabular_data - self.tabular_mean) / self.tabular_std
        
        tabular_data = process_tabular_data(tabular_data)

     
        data_transform = T.Compose([
            T.ToTensor(),
            T.CenterCrop((self.resolution[0], self.resolution[1])) if self.resolution is not None else nn.Identity(),
            # T.Resize((self.resolution[0], self.resolution[1]), antialias=True) if self.resolution is not None else nn.Identity(),
            T.RandomVerticalFlip() if self.random_flip else nn.Identity(),
            T.RandomAffine(180, translate=(0.3, 0.3)) if self.random_affine else nn.Identity(),         
        ])

        processed_mri = np.zeros((self.output_dim, self.resolution[0], self.resolution[1])).astype(np.float32)
        processed_pet = np.zeros((self.output_dim, self.resolution[0], self.resolution[1])).astype(np.float32)
        processed_loss_weight_mask = np.zeros((self.output_dim, self.resolution[0], self.resolution[1])).astype(np.float32)

        for i in range(mri_scan.shape[0]):
            _mri_data = mri_scan[i, :, :]
            _pet_data = pet_scan[i, :, :]

            _loss_weight_mask = loss_weight_mask[i, :, :]

            _mri_data = data_minimal_process(self.resolution, _mri_data, data_transform)
            _pet_data = data_minimal_process(self.resolution, _pet_data, data_transform)
            _loss_weight_mask = data_minimal_process(self.resolution, _loss_weight_mask, data_transform)

            processed_mri[i, :, :] = _mri_data
            processed_pet[i, :, :] = _pet_data
            processed_loss_weight_mask[i, :, :] = _loss_weight_mask
        
        mri_scan = processed_mri
        pet_scan = processed_pet
        loss_weight_mask = processed_loss_weight_mask


        label = self._diagnosis[idx]
        mri_uid = self._mri_uid[idx]


        assert mri_scan.shape == (self.output_dim, self.resolution[0], self.resolution[1]), 'mri scan shape is not correct'
        assert pet_scan.shape == (self.output_dim, self.resolution[0], self.resolution[1]), 'pet scan shape is not correct'
   
        return mri_scan, pet_scan, label, tabular_data, slice_index, loss_weight_mask, mri_uid