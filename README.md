# PDC-Depth

This is the PyTorch implementation for **"Label-Free Adaptation of Video Foundation Models to Endoscopic Depth via Progressive Trustworthy Knowledge Distillation Chain."**

[Project Page](https://pdc-depth.github.io/)

## ⚙️ Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate pdc-depth
```

Download the two initialization checkpoints from [Google Drive](https://drive.google.com/drive/folders/1UwEscB8lVHxHEzaM_9lSfK5-uOAuacvc?usp=sharing) and place them in `pretrained_model/`:

- `video_depth_anything_vits.pth`
- `depth_anything_vitb14.pth`

## 📁 Data Preparation

Please download the datasets from their official pages:

- [SCARED](https://endovissub2019-scared.grand-challenge.org/)
- [Hamlyn](https://davidrecasens.github.io/EndoDepthAndMotion/)
- [SimCol3D](https://github.com/anitarau/simcol)
- [C3VD](https://github.com/DurrLab/C3VD)

The exact data splits used in our experiments are provided in `splits/`.

| Dataset | Split name | Training frames | Test frames | Maximum depth |
|---|---:|---:|---:|---:|
| SCARED | `endovis` | 15,351 | 551 | 150 mm |
| Hamlyn | `hamlyn` | 16,841 | 8,063 | 300 mm |
| SimCol3D | `simcol` | 28,776 | 9,009 | 200 mm |
| C3VD | `c3vd` | 8,526 | 1,171 | 100 mm |

## 🚀 Evaluation

### Model Zoo

| Training dataset | Evaluation datasets | Model weight |
|---|---|---|
| SCARED | SCARED, Hamlyn | [Download](SCARED_MODEL_LINK_TO_BE_ADDED) |
| Hamlyn | Hamlyn, SCARED | [Download](HAMLYN_MODEL_LINK_TO_BE_ADDED) |
| SimCol3D | SimCol3D, C3VD | [Download](SIMCOL3D_MODEL_LINK_TO_BE_ADDED) |
| C3VD | C3VD, SimCol3D | [Download](C3VD_MODEL_LINK_TO_BE_ADDED) |

### Depth Evaluation

Run depth evaluation with the corresponding dataset path, model weights, split, maximum depth, and input size:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py \
  --data_path <DATA_PATH> \
  --load_weights_folder <WEIGHTS_FOLDER> \
  --eval_split <SPLIT_NAME> \
  --max_depth <MAX_DEPTH> \
  --height <INPUT_HEIGHT> --width <INPUT_WIDTH> \
  --num_frames 6 --batch_size 1 --eval_mono
```

Add `--png` when the dataset images are stored in PNG format.

## 📊 Results

### Main Results

| Training set | Test set | Abs Rel | Sq Rel | RMSE | RMSE log | δ |
|---|---|---:|---:|---:|---:|---:|
| SCARED | SCARED | 0.046 | 0.298 | 4.063 | 0.066 | 0.984 |
| SCARED | Hamlyn | 0.048 | 0.616 | 6.421 | 0.071 | 0.984 |
| SimCol3D | SimCol3D | 0.056 | 0.026 | 0.337 | 0.086 | 0.977 |
| SimCol3D | C3VD | 0.089 | 0.360 | 3.322 | 0.110 | 0.929 |

### Additional Results

| Training set | Test set | Abs Rel | Sq Rel | RMSE | RMSE log | δ |
|---|---|---:|---:|---:|---:|---:|
| Hamlyn | Hamlyn | 0.052 | 0.647 | 6.668 | 0.075 | 0.984 |
| C3VD | C3VD | 0.127 | 0.731 | 5.061 | 0.155 | 0.829 |
| Hamlyn | SCARED | 0.057 | 0.445 | 4.986 | 0.081 | 0.974 |
| C3VD | SimCol3D | 0.222 | 0.271 | 1.062 | 0.288 | 0.632 |

## 📦 Release Status

- ✅ **Evaluation code and data splits** are available.
- ✅ **Model weights** are available.
- ⏳ **Training code** will be released upon acceptance.

## 🙏 Acknowledgements

This project is built upon [EndoDAC](https://github.com/BeileiCui/EndoDAC), [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything), [Depth Anything](https://github.com/LiheYoung/Depth-Anything), and [Monodepth2](https://github.com/nianticlabs/monodepth2). We thank the authors for their excellent works.
