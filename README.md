# Meta Feature Disentanglement under Continuous-Valued Domain Modeling for Generalizable Remote Sensing Image Segmentation on Unseen Domains

This is the official implementation of the ISPRS-JPRS paper:**"Meta Feature Disentanglement under Continuous-Valued Domain Modeling for Generalizable Remote Sensing Image Segmentation on Unseen Domains"**
![MetaFD](https://github.com/user-attachments/assets/655797b8-8bc8-423e-8932-7c3a505001fd)
![meta-training](https://github.com/user-attachments/assets/75123968-22f9-4cc2-9622-d056025249a0)

## 📌 Introduction
This project proposes a meta-learning-based domain generalization method for remote sensing image segmentation, which disentangles domain-invariant and domain-specific features to improve generalization on unseen domains.

## 🛠 Environment Setup
We recommend using Python 3.11+. Install dependencies with:
>pip install -r requirements.txt

## 📁 Data Preparation
Organize your datasets in the following structure (Refer to "data" folder):
```bash
data/
├── CITY-OSM/
│   ├── CO-PA/
│   │   ├── sample/
│   │   │   ├── train/
│   │   │   │   ├── image/
│   │   │   │   ├── label/
│   │   │   └── val/
│   │   │   │   ├── image/
│   │   │   │   ├── label/
│   │   ├── result/
│   │   └── weight/
└── [others]/
```
Refer to Section 4.1 of the paper for dataset details and preparation.

## 🚀 Usage
### Training
```bash
python trainer.py \
    --gpu_number 0 \
    --data_dir "/path/to/your/data" \
    --dg_type "MDG" \
    --target_domain "CITY-OSM/CO-PA" \
    --epoch 100 \
    --batch_size 6 \
    --learning_rate 1e-4 \
    --seg_model "UNet" \
    --backbone "vgg16" \
    --classes 3
```

### Inference
```bash
python predictor.py \
    --gpu_number 0 \
    --td "CITY-OSM/CO-PA" \
    --data_dir "/path/to/your/data" \
    --weight_path "/path/to/model/weights.pth" \
    --dg_type "SDG" \
    --bs 1 \
    --classes 3
```

## 📜 Citation
If you use this code in your research, please cite our paper:
```bash
@article{liang2025meta,
title = {Meta Feature Disentanglement under continuous-valued domain modeling for generalizable remote sensing image segmentation on unseen domains},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {230},
pages = {738-753},
year = {2025},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2025.09.029},
url = {https://www.sciencedirect.com/science/article/pii/S0924271625003879},
author = {Chenbin Liang and Xiaoping Zhang and Wenlin Fu and Weibin Li and Yunyun Dong},
}
```