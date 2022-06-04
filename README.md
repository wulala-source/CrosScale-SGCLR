# CrosScale-SGCLR

传统基于人体骨架的自监督学习算法常用对比学习方法进行表征学习，而现有对比学习方法使用数据增强方法来构造相似的正样本，其余样本皆为负样本，这限制了同类样本的信息表达。针对上述问题，提出一种基于图对比学习与跨尺度一致性知识挖掘的动作识别算法。首先，基于骨架图结构设计了一种新的数据增强方法，对输入的骨架序列进行随机边裁剪，得到两个不同的视图，从而增强同一骨架序列不同视图间的语义相关性表达；其次，为缓解同类样本嵌入相似度较低的问题，引入自监督协同训练网络模型，利用同一骨架数据源的不同尺度间的互补信息，从一个骨架尺度获取另一个骨架尺度的正类样本，实现单尺度内关联及多尺度间语义协同交互；最后，基于线性评估协议对模型效果进行评估。

## Requirements
  环境配置:
  - Python == 3.6.0
  - PyTorch == 1.2.0
  - CUDA == 9.2

## Installation
  ```bash
  # Install torchlight
  $ cd torchlight
  $ python setup.py install
  $ cd ..
  # Install other python libraries
  $ pip install -r requirements.txt
  ```

## Data Preparation
- 下载数据集 [NTU RGB+D](https://github.com/shahroudy/NTURGB-D) 
- python tools/ntu60_gendata.py 
- 获取50帧骨架序列：python feeder/preprocess_ntu.py

## Unsupervised Pre-Training
```bash
# train on NTU RGB+D 60 xview joint stream
$ python main.py pretrain_sgclr --config config/ntu60/pretext/pretext_sgclr_xview_joint.yaml

# train on NTU RGB+D 60 xview joint stream
$ python main.py pretrain_crosview_sgclr --config config/ntu60/pretext/pretext_crosview_sgclr_xview_joint.yaml

# train on NTU RGB+D 60 xview joint stream
$ python main.py pretrain_crosscale_sgclr --config config/ntu60/pretext/pretext_crosscale_sgclr_xview_joint.yaml
```

## Linear Evaluation
```bash
# Linear_eval on NTU RGB+D 60 xview
$ python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_sgclr_xview_joint.yaml

$ python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_crosview_sgclr_xview_joint.yaml

$ python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_crosscale_sgclr_xview_joint.yaml
```
  
## Acknowledgement
- 本文代码框架是从以下论文代码中扩展的，非常感谢作者们发布这些代码。
- 代码框架是基于 [CrosSCLR](https://github.com/LinguoLi/CrosSCLR).
- 编码器网络基于 [ST-GCN](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md).
