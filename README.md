# Pathology-Aware MRI to PET Cross-modal Translation with Diffusion Models
Official Pytorch Implementation of Paper - 🍝 [PASTA: Pathology-Aware MRI to PET Cross-modal Translation with Diffusion Models](https://arxiv.org/abs/2405.16942)

[![Conference Paper](https://img.shields.io/static/v1?label=DOI&message=10.1007%2f978-3-031-34048-2_51&color=3a7ebb)](https://doi.org/10.1007/978-3-031-72104-5_51)
[![Preprint](https://img.shields.io/badge/arXiv-2303.07717-b31b1b)](https://arxiv.org/abs/2405.16942)

🎉 PASTA has been early-accepted at [MICCAI 2024](https://conferences.miccai.org/2024/en/) (top 11%)!

<p align="center">
  <img src="img/pasta.png" />
</p>


## Installation

1. Create environment: `conda env create -n pasta --file requirements.yaml`
2. Activate environment: `conda activate pasta`


## Data

We used data from [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/). Since we are not allowed to share our data, you would need to process the data yourself. Data for training, validation, and testing should be stored in separate [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) files, using the following hierarchical format:

1. First level: A unique identifier, e.g. image ID.
2. The second level always has the following entries:
    1. A group named `MRI/T1`, containing the T1-weighted 3D MRI data.
    2. A group named `PET/FDG`, containing the 3D FDG PET data.
    3. A dataset named `tabular` of size 6, containing a list of non-image clinical data, including age, gender, education, MMSE, ADAS-Cog-13, ApoE4.
    4. A string attribute `DX` containing the diagnosis labels: `CN`, `Dementia` or `MCI`, if available.
    5. A scalar attribute `RID` with the patient ID, if available.
    6. A string attribute `VISCODE` with ADNI's visit code.

Finally, the HDF5 file should also contain the following meta-information in a separate group named `stats`:
```bash
/stats/tabular           Group
/stats/tabular/columns   Dataset {6}
/stats/tabular/mean      Dataset {6}
/stats/tabular/stddev    Dataset {6}
```
They are the names of the features in the tabular data, their mean, and standard deviation.


## Usage

The package uses [PyTorch](https://pytorch.org). To train and test PASTA, execute the `train_mri2pet.py` script. 
The configuration file of the command arguments is stored in `src/config/pasta_mri2pet.yaml`.
The essential command line arguments are:

  - `--data_dir`: Path prefix to HDF5 files containing either train, validation, or test data.
  - `--results_folder`: Path to save all the training/testing output.
  - `--model_cycling`: *True* to conduct cycle exchange consistency.
  - `--eval_mode`: *False* for training mode and *True* for evaluation mode. The model for evaluation is specified in `results_folder/model.pt`.
  - `--synthesis`: *True* to save all generated images during evaluation.


After specifying the config file, simply start training/evaluation by:
```bash
python train_mri2pet.py
```

## Contacts

For any questions, please contact: Yitong Li (yi_tong.li@tum.de)

## Acknowlegements

The codebase is developed based on [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) and [openai/guided-diffusion](https://github.com/openai/guided-diffusion).

If you find this repository useful, please consider giving a star 🌟 and citing the paper:

```bibtex
@InProceedings{Li2024pasta,
    author="Li, Yitong
    and Yakushev, Igor
    and Hedderich, Dennis M.
    and Wachinger, Christian",
    title="PASTA: Pathology-Aware MRI to PET Cross-Modal Translation with Diffusion Models",
    booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024",
    year="2024",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="529--540",
    isbn="978-3-031-72104-5"
}
```
