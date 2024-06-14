
<p align="center">
    <img src="git-imgs/single_drug_img.png" width="700" style="margin-bottom: 0.2;"/>
<p>
<h2 align="center"> <a href="">Deep peak property learning for efficient chiral molecules ECD spectra prediction</a></h2>
<h5 align="center"> The official code for "Deep peak property learning for efficient chiral molecules ECD spectra prediction" submitted to Nature Machine Intelligence. Here we publish the inference code of ECDFormer. The training code & ECD spectra dataset will be released after our paper is accepted. If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>


<h5 align="center">
    
[![arXiv](https://img.shields.io/badge/Arxiv-2310.01852-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2401.03403)
[![License](https://img.shields.io/badge/Code%20License-MIT-yellow)](https://github.com/HowardLi1984/ECDFormer/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Dataset%20license-CC--BY--NC%204.0-orange)](https://github.com/HowardLi1984/ECDFormer/blob/main/DATASET_LICENSE) <br>

</h5>

## Data Preparation
For training and inference, please download and put the [descriptor_all_column.npy](https://drive.google.com/file/d/1MHRkm4Jp4SBafwSFXyxsh1H2UdE2cEDc/view?usp=sharing) into the folder utils/
```bash
utils/descriptor_all_column.npy
```
We will release the CMCDS dataset for training procedure once our paper is accepted.

## 🛠️ Requirements and Installation
* Python == 3.8
* Pytorch == 1.13.1
* CUDA Version == 11.7
* torch_geometric, troch-scatter, torch-sparse, torch-cluster, torch-spline-conv
* Install required packages:
```bash
git clone git@github.com:HowardLi1984/ECDFormer.git
cd ECDFormer
pip install -r requirements.txt
```

## 🗝️ Inferencing
The inferencing instruction is in [main_func_pos.py](main_func_pos.py).
```bash
CUDA_VISIBLE_DEVICES=0 python main_func_pos.py --model_name gnn_allthree --batch_size 256 --emb_dim 128 --epochs 1000 --lr 1e-3 --mode Real --visual_epoch 400
```

## ✏️ Citation
If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.

```BibTeX
@article{li2024deep,
  title={Deep peak property learning for efficient chiral molecules ECD spectra prediction},
  author={Li, Hao and Long, Da and Yuan, Li and Tian, Yonghong and Wang, Xinchang and Mo, Fanyang},
  journal={arXiv preprint arXiv:2401.03403},
  year={2024}
}
```