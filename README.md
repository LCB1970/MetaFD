# Meta Feature Disentanglement under Continuous-Valued Domain Modeling for Generalizable Remote Sensing Image Segmentation on Unseen Domains

This is the official implementation of the ISPRS-JPRS paper:**"Meta Feature Disentanglement under Continuous-Valued Domain Modeling for Generalizable Remote Sensing Image Segmentation on Unseen Domains"**

![MetaFD.jpg](..%2FUsers%2F34466%2FDesktop%2FBaiduSyncdisk%2FMetaFD%2Fmanuscript%2Fimg%2FMetaFD.jpg)
![meta-training.jpg](..%2FUsers%2F34466%2FDesktop%2FBaiduSyncdisk%2FMetaFD%2Fmanuscript%2Fimg%2Fmeta-training.jpg)

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
  title={Meta Feature Disentanglement under Continuous-Valued Domain Modeling for Generalizable Remote Sensing Image Segmentation on Unseen Domains},
  author={Chenbin Liang, Xiaoping Zhang, Wenlin Fu, Weibin Li and Yunyun Dong},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  year={2025}
}
```