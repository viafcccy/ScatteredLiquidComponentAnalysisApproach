# ScatteredLiquidComponentAnalysisApproach



此仓库为论文 [T. Liu, C. Zhou, C. Fang, H. Zhu, Y. Li and J. Wu, "A Scattered Liquid Component Analysis Approach Based on Spectral Visual Encoding and Fusion," in IEEE Sensors Journal, doi: 10.1109/JSEN.2023.3336797.](https://ieeexplore.ieee.org/document/10341277) 源码



## 项目背景



主要提出了一种从多谱线特征层融合角度提出了基于格拉姆角场（Gramian angular field）与多尺度卷积神经网络（multi-scale convolutional neural network）的光谱特征提取方法，使用格拉姆角场将一维光谱信号变换为二维图像增强了信号特征，使用多尺度卷积神经网络提取图像特征，利用特征向量回归计算金属离子浓度。



配合./doc中的中文论文原稿和[T. Liu, C. Zhou, C. Fang, H. Zhu, Y. Li and J. Wu, "A Scattered Liquid Component Analysis Approach Based on Spectral Visual Encoding and Fusion," in IEEE Sensors Journal, doi: 10.1109/JSEN.2023.3336797.](https://ieeexplore.ieee.org/document/10341277) 正式发表论文参考



## 目录结构

### ./data

- 此目录主要存放原始数据和各种处理后的数据
- 以及各种处理脚本
- process_2d_new.py 为批量将目标库中的csv转为格拉姆角场（CNN训练中只使用目标库数据）
  ref_2d_process.py 处理参考库中的csv转为格拉姆（对比测试用，实际训练中未使用）

### ./cnn

- 此目录存放神经网络代码

### ./gaf_show

- 此目录存放格拉姆角场可视化文件

### ./pretreatment

- 此目录下为光谱矫正和原始光谱误差等预处理文件







