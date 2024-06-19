# Mask R-CNN
## 环境配置：
* Python3.6/3.7/3.8
* Pytorch1.10或以上
* 最好使用GPU训练
* 详细环境配置见`requirements.txt`
## Demo Video
https://github.com/chengxixugong/Image-Segmentation/blob/master/6%E6%9C%8819%E6%97%A5.mp4


## 文件结构：
```
  ├── backbone: 特征提取网络
  ├── network_files: Mask R-CNN网络
  ├── train_utils: 训练验证相关模块（包括coco验证相关）
  ├── my_dataset_coco.py: 自定义dataset用于读取COCO2017数据集
  ├── my_dataset_voc.py: 自定义dataset用于读取Pascal VOC数据集
  ├── train.py: 单GPU/CPU训练脚本
  ├── train_multi_GPU.py: 针对使用多GPU的用户使用
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测
  ├── validation.py: 利用训练好的权重验证/测试数据的COCO指标，并生成record_mAP.txt文件
  └── transforms.py: 数据预处理（随机水平翻转图像以及bboxes、将PIL图像转为Tensor）
```


### Pascal VOC2012数据集
* 数据集下载地址： http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
* 解压后得到的文件夹结构如下：
```
VOCdevkit
    └── VOC2012
         ├── Annotations               所有的图像标注信息(XML文件)
         ├── ImageSets
         │   ├── Action                人的行为动作图像信息
         │   ├── Layout                人的各个部位图像信息
         │   │
         │   ├── Main                  目标检测分类图像信息
         │   │     ├── train.txt       训练集(5717)
         │   │     ├── val.txt         验证集(5823)
         │   │     └── trainval.txt    训练集+验证集(11540)
         │   │
         │   └── Segmentation          目标分割图像信息
         │         ├── train.txt       训练集(1464)
         │         ├── val.txt         验证集(1449)
         │         └── trainval.txt    训练集+验证集(2913)
         │
         ├── JPEGImages                所有图像文件
         ├── SegmentationClass         语义分割png图（基于类别）
         └── SegmentationObject        实例分割png图（基于目标）
```

