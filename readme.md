**BraTS-UMamba: Adaptive Mamba UNet with Dual-Band Frequency based Feature Enhancement for Brain Tumor Segmentation**
==============================
We have now open-sourced the code for data preprocessing, model training, inference, and evaluation metric computation.
The link to our paper:


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
	


