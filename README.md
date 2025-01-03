# JAG-MdsNet
This is the implementation of the article: "**A Multi-Domain Dual-Stream Network for Hyperspectral Unmixing**".  

**"Results_samson.mat","Results_jasper.mat","Results_urban.mat":**  
The mats contain all the source unmixing results of the **MdsNet**. 
   >**A** denotes the unmixing results of abundance maps;  
    **M1** denotes the unmixing results of endmembers;  
    **Y_c** denotes the coarse domain pixel data we conducted in the experiments;  
    **Y_d** denotes the differential pixels of the datasets, which can be expressed as Yc = Y - Y_d;  

## Implement of MdsNet

### Configuration requirements 

1. Python  3.9.13
2. Pytorch 1.11.0

### Dataset
All datasets used in this work can be found in [Google Driver](https://drive.google.com/drive/folders/1Tfj7371mOVatDI4vRcG2O7Xk2lNKW05C?usp=drive_link).
These datasets can be directly used to reproduce the results presented in the article.

### Usage

Run ``MdsNet_japser_demo.py``

The format of input:
  >**A** denotes the ground truth of abundance;  
   **M** denotes the ground truth of endmembers;  
   **M1** denotes the initialization of endmembers by VCA;  
   **Y_c** denotes the coarse domain pixel data we conducted in the experiments;  
   **Y** denotes the HSI;  
