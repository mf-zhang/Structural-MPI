# Structural MPI

(CVPR2023) Structural Multiplane Image: Bridging Neural View Synthesis and 3D Reconstruction  [Paper](https://arxiv.org/abs/2303.05937) 

We propose the Structural MPI (S-MPI) framework, where we assign an MPI layer to each 3D surface in the scene. Each MPI layer in our S-MPI is adaptively posed based on planar 3D reconstruction of the scene. In contrast, standard MPI is composed of a set of layers that are all parallel to the imaging plane.

This project aims to construct an S-MPI from one or several images while conducting planar 3D reconstruction and novel view synthesis.

## Installation

```
conda create -n SMPI python=3.7
conda activate SMPI
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
cd detectron2
pip install -e . 
cd ../Mask2Former
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
pip install opencv-python piqa moviepy
```

This project is built based on [Mask2former](https://github.com/facebookresearch/Mask2Former). If you find any problem in the installation, [this page](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md) should be helpful.

## Data Preparation

Please download ScanNetv2 http://www.scan-net.org and download plane annotations https://drive.google.com/file/d/1B2Y0-bfsmjvrMfrTMLhu_gn0wYfl3pIS/view?usp=sharing

The files should be arranged like

```
/path/to/ScanNet/scans/scan_plane/
/path/to/ScanNet/scans/scene0000_00/
...
```

## Checkpoints

Our trained checkpoints can be downloaded at https://drive.google.com/drive/folders/16qunh66bGzie5S9eq0KszIO6kH6pblre?usp=sharing where `input1` was trained with single-view input and `input2` with multi-view input.

## Run a test

Please run `sh run_test.sh` and the output will be printed and saved in `./testing_output/` which can be visualized by tensorboard.

## Reference

https://github.com/facebookresearch/Mask2Former

https://github.com/vincentfung13/MINE

https://github.com/IceTTTb/PlaneTR3D

https://github.com/EryiXie/PlaneRecNet

## Citation
```
@inproceedings{zhang2023structural,
  title={Structural Multiplane Image: Bridging Neural View Synthesis and 3D Reconstruction},
  author={Zhang, Mingfang and Wang, Jinglu and Li, Xiao and Huang, Yifei and Sato, Yoichi and Lu, Yan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16707--16716},
  year={2023}
}
```