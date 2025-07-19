**BraTS-UMamba: Adaptive Mamba UNet with Dual-Band Frequency based Feature Enhancement for Brain Tumor Segmentation**
==============================
This is the official PyTorch implementatin of our project, BraTS-UMamba, which was early accepted by MICCAI 2025.

## Project description
- we propose BraTS-UMamba, a novel Mamba-based U-Net designed to enhance brain tumor segmentation by capturing and adaptively fusing bi-granularity based long-range dependencies in the spatial domain while integrating both low- and high-band spectrum clues from the frequency domain to refine spatial feature representation.
- We further enhance segmentation through an auxiliary brain tumor classification loss.
- Extensive experiments on two public benchmark datasets demonstrate the superiority of our BraTS-UMamba over state-of-the-art methods.

## Network architecture
<img width="2087" height="1247" alt="image" src="https://github.com/user-attachments/assets/afe0b265-9892-4d7a-9cf4-94784089e7c3" />


## Experimental results
### Comparative results on MSD BTS
<img width="2078" height="708" alt="image" src="https://github.com/user-attachments/assets/837c2a55-4b62-482e-a57c-649b75f78b7c" />
### Comparative results on BraTS2023-GLI
<img width="2088" height="700" alt="image" src="https://github.com/user-attachments/assets/8f09006b-7f55-4d03-a1bf-c5f5e82c4c4d" />


## Bibtex entry to our paper
-------
	@article{yao2025brats,
    title={BraTS-UMamba: Adaptive Mamba UNet with Dual-Band Frequency Based Feature Enhancement for Brain Tumor Segmentation},
    author={Haoran Yao and Hao Xiong and Dong Liu and Hualei Shen and Shlomo Berkovsky},
    journal={MICCAI 2025},
    year={2025}
Environment Configuration
------
	# CUDA 11.8
	pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

	#Install Mamba
	causal-conv1d==1.0.0
	mamba-ssm==1.0.1
Data Preprocessing
---
	bash generate_hdf5_record_from_directory_MSD.bash
Training
---
	bash run.bash
Trained model weights
------
&emsp;Click the link to download:[Trained model](https://pan.baidu.com/s/1Uj8qfArXeBbKsogRDyvkuA?pwd=6666)

Inference
---
	bash test_eval.bash
