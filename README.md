# Mitigating Noisy Correspondence by Geometrical Structure Consistency Learning

<div align="center">   <a href="https://arxiv.org/abs/2405.16996">     <img src="https://img.shields.io/badge/arXiv-2405.16996-b31b1b" alt="arXiv">   </a>   <a href="https://openreview.net/forum?id=lgZf628A2G&referrer=%5Bthe%20profile%20of%20Zihua%20Zhao%5D(%2Fprofile%3Fid%3D~Zihua_Zhao3)">     <img src="https://img.shields.io/badge/OpenReview-CVPR24-blue" alt="OpenReview">   </a>  
  <a href="https://github.com/MediaBrain-SJTU/GSC">     <img src="https://img.shields.io/badge/GitHub-GSC-brightgreen" alt="GitHub">   </a> </div>

by Zihua Zhao, Mengzi Chen, Tianjie Dai, Jiangchao Yao, Bo Han, Ya Zhang, Yanfeng Wang at Cooperative Medianet Innovation Center, Shanghai Jiao Tong University, Shanghai Artificial Intelligence Laboratory, Hong Kong Baptist University

Computer Vision and Pattern Recognition (CVPR), 2024.

This repo is the official Pytorch implementation of GSC.

⚠️ This repository is being organized and updated continuously. Please note that this version is not the final release.

## Environment

Code is implemented based on original code provided by DECL from https://github.com/QinYang79/DECL, which offers the standard code for retrieval framework, data loader and evaluation metrics.

Besides, create the environment for running our code:

```bash
conda create --name GSC python=3.9.13
conda activate GSC
pip install -r requirements.txt
```

## Data Preparation

For all of the dataset settings, we follow DECL including data split, sample preprocessing and noise simulation.

#### MS-COCO and Flickr30K

MS-COCO and Flickr30K are two datasets with simulated noisy correspondence. The simulated noise is created by randomly flipping captions. The images in the datasets are first extracted into features by SCAN (https://github.com/kuanghuei/SCAN).

### CC152K

CC152K is a real-world noisy correspondence dataset. The images are also extracted into features by SCAN.

Notably, SCAN is implemented on Python2.7 and Caffe, which may not be supported on latest Cuda versions. We are extracting the features through renting machines from platform AutoDL (https://www.autodl.com/login?url=/console/homepage/personal).

## Running 

For training on different datasets, we provide integrated bash orders. Evaluation metrics are implemented in eval.py.

```
# for training and evaluation on Flickr30K as an example
conda activate GSC
sh train_f30k.sh
python eval.py
```

## Citation

If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.

```
@inproceedings{zhao2024mitigating,
  title={Mitigating Noisy Correspondence by Geometrical Structure Consistency Learning},
  author={Zhao, Zihua and Chen, Mengxi and Dai, Tianjie and Yao, Jiangchao and Han, Bo and Zhang, Ya and Wang, Yanfeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={27381--27390},
  year={2024}
}
```

If you have any problem with this code, please feel free to contact **[sjtuszzh@sjtu.edu.cn](mailto:sjtuszzh@sjtu.edu.cn)**.