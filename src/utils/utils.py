import math
import torch
import argparse
import yaml
import random
import numpy as np

# helpers functions

def load_config_from_yaml(config_file_path):
    with open(config_file_path, 'r') as f:
        config_data = yaml.safe_load(f)

    args = argparse.Namespace(**config_data)
    return args

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def cycle_pair(dl):
    while True:
        for mri_data, pet_data in dl:
            yield mri_data, pet_data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor): #(4,8)
    groups = num // divisor # 0
    remainder = num % divisor # 4
    arr = [divisor] * groups # [0]
    if remainder > 0:
        arr.append(remainder) # [0,4]
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def print_model_size(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

def set_seed_everywhere(seed):
    """Set seed for reproducibility."""
    # Python
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def positional_encoding_2d(x, h, w):
    B = x.shape[0]
    
    # Generate row and column indices
    row_indices = torch.linspace(0, 1, h).view(1, -1, 1).expand(B, -1, -1).to(x.device)
    col_indices = torch.linspace(0, 1, w).view(1, 1, -1).expand(B, -1, -1).to(x.device)
    
    # Reshape x for broadcasting
    x = x.view(B, 1, 1)
    
    # Create sinusoidal patterns modulated by the input value batch
    row_pattern = torch.sin(2 * 3.14159 * x * 10 * row_indices)
    col_pattern = torch.sin(2 * 3.14159 * x * 10 * col_indices)

    # Combine row and column patterns
    encoded = row_pattern + col_pattern

    return encoded
