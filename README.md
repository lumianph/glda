# GLDA: GPU-Accelerated Latent Dirichlet Allocation Training

## Introduction
GLDA (GPU-accelerated LDA) is an improved version based on the popular Latent Dirichlet Allocation (LDA) software GibbsLDA++: http://gibbslda.sourceforge.net/ The same user interfaces are adopted in GLDA. However, the training speed is singificantly improved. For the current version with one GPU used, a speedup of around 15X is achieved compared with the original CPU-based GibbsLDA++.

## INSTALL
Please modify the Makefile to specific the compute capability of your NVIDIA GPU cards. Then just type: **make**

Then an executable file *lda* is generated in the directory.

## Citation
If you would like cite this work:

**Mian Lu, Ge Bai, Qiong Luo, Jie Tang, Jiuxin Zhao. Accelerating Topic Model Training on a Single Machine. APWeb 2013: 184-195**

## Contact
Mian Lu
lumianph@gmail.com