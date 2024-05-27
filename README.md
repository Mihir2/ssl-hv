# SSL-HV: Self Supervised Learning based Handwriting Verification
This repository provides a PyTorch Lightning implementation and Pretrained models for SSL-HV, as described in the paper SSL-HV: Self Supervised Learning based Handwriting Verification <br>
_Mihir Chauhan, Mohammad Abuzar Shaikh, Bina Ramamurthy, Mingchen Gao, Siwei Lyu, Sargur Srihari_ <br>
_The State Unviersity of New York at Buffalo, USA_ <br>
## CEDAR Handwriting Dataset
| Dataset | Link to Drive |
|:-----------|:------------:|
| CEDAR Letter Images | [Link](https://drive.google.com/drive/folders/1fwRlwtfzV_5Pnyxm9ahQLi2eum2rDshk?usp=sharing)  |
| CEDAR AND Images| [Link](https://drive.google.com/drive/folders/1uj6eeaKBmabivxvRqrGokrcCb3B9yAHu?usp=sharing)  |
| CEDAR AND GSC Features | [Link](https://drive.google.com/drive/folders/1sqKDswK-w2elL8uuJD0HdqlBZNd1hvFG?usp=sharing) |

## SL-HV: Supervised Handwriting Verification (Baseline)
| Model                                 | Accuracy  | Precision | Recall    | F1-Score |  Model |
|:--------|:----------:|:----------:|:----------:|:----------:|:----------:|
| GSC             | 0.71 / 0.78 | 0.69 / 0.81 | 0.72 / 0.77 | 0.69 / 0.79 | [Link]() |
| ResNet-18 | 0.72 / 0.84 | 0.70 / 0.86 | 0.73 / 0.82 | 0.72 / 0.84 | [Link]() |
| ViT | 0.65 / 0.79 | 0.68 / 0.80 | 0.64 / 0.78 | 0.66 / 0.79 | [Link]() |

## CSSL-HV: Contrastive Self-Supervised Handwriting Verification
| Model  | Intra-Nd | Inter-Nd | Intra-2d | Inter-2d | Accuracy |  Model |
|:--------|:----------:|:----------:|:----------:|:----------:|:----------:| :----------:|
| Raw Pixels | 0.96     | 0.95     | 0.07     | -0.02    | 0.63     |[Link]() |
| HOGS  | 0.57     | 0.02     | 0.63     | 0.11     | 0.72     |[Link]() |
| GSC   | 0.92     | 0.67     | 0.86     | 0.56     | 0.71     |[Link]() |
| AIM  | 0.32     | -0.05    | 0.78     | 0.75     | 0.73     |[Link]() |
| Flow | 0.12 | 0.08 | 0.12 | 0.01 | 0.66 |[Link]() |
| MAE  | 0.18     | 0.02     | 0.82     | 0.77     | 0.71     |[Link]() |
| **VAE**  | 0.24 | 0.06 | 0.38 | 0.30 | **0.75** |[Link]() |

## GSSL-HV: Contrastive Self-Supervised Handwriting Verification
| Model  | Intra-Nd | Inter-Nd | Intra-2d | Inter-2d | Accuracy |  Model |
|:--------|:----------:|:----------:|:----------:|:----------:|:----------:| :----------:|
| BiGAN  | 0.35 | 0.30 | 0.27 | 0.25 | 0.68 |[Link]() |
| MoCo  | 0.89 | 0.78 | 0.92 | 0.73 | 0.73 |[Link]() |
| SimClr | 0.89 | 0.87 | 0.87 | 0.85 | 0.72 |[Link]() |
| BYOL  | 0.88 | 0.84 | 0.91 | 0.97 | 0.73 |[Link]() |
| SimSiam  | 0.87 | 0.81 | 0.94 | 0.84 | 0.75 |[Link]() |
| FastSiam  | 0.83 | 0.75 | 0.83 | 0.75 | 0.71 |[Link]() |
| DINO | 0.88 | 0.85 | 0.78 | 0.74 | 0.68 |[Link]() |
| BarlowTwins | 0.87 | 0.79 | 0.66 | 0.38 | 0.76 |[Link]() |
| **VicReg** | 0.69 | 0.48 | 0.65 | 0.60 | **0.78** |[Link]() |

## Cite
[SSL-HV paper](https://arxiv.org/): (Pending Arxiv Moderation Id: 5617299)
```
@article{5617299,
  title={Self-Supervised based Handwriting Verification},
  author={Mihir Chauhan, Mohammad Abuzar Shaikh, Bina Ramamurthy, Mingchen Gao, Siwei Lyu and Sargur Srihari},
  journal={arXiv preprint arXiv:},
  year={2024}
}
```
