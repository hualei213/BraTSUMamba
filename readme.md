**BraTS-UMamba: Adaptive Mamba UNet with Dual-Band Frequency based Feature Enhancement for Brain Tumor Segmentation**
==============================
This is the official PyTorch implementatin of our project, BraTS-UMamba, which was early accepted by MICCAI 2025.

# Project description
- we propose BraTS-UMamba, a novel Mamba-based U-Net designed to enhance brain tumor segmentation by capturing and adaptively fusing bi-granularity based long-range dependencies in the spatial domain while integrating both low- and high-band spectrum clues from the frequency domain to refine spatial feature representation.
- We further enhance segmentation through an auxiliary brain tumor classification loss.
- Extensive experiments on two public benchmark datasets demonstrate the superiority of our BraTS-UMamba over state-of-the-art methods.

# Network Architecture
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/298dc9b986454c26b359ede72dddd54e.png#pic_center)


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/51d33d00a482434d9fc1f41debccd1a4.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/af55ddf1d8984a9d85ce80c92c4f461b.png#pic_center)
Paper
-------
	@article{yao2025brats,
    title={BraTS-UMamba: Adaptive Mamba UNet with Dual-Band Frequency Based Feature Enhancement for Brain Tumor Segmentation},
    author={Haoran Yao and Hao Xiong and Dong Liu and Hualei Shen and Shlomo Berkovsky},
    journal={},
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
