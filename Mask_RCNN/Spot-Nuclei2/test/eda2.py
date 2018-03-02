'''
为什么val看起来很好，test不太行？

1. 计算训练集和测试集的均值
2. 训练集和测试集尺寸大小的差异？
3. 训练集和测试集分别有几种？

https://www.kaggle.com/kmader/normalizing-brightfield-stained-and-fluorescence/notebook 有用吗

https://www.kaggle.com/ramzes2/distribution-of-nuclei-sizes
https://www.kaggle.com/pudae81/data-visualization-and-analysis
https://www.kaggle.com/etheleon/exploratory-analysis-image-stats
https://www.kaggle.com/mpware/stage1-eda-microscope-image-types-clustering
https://www.kaggle.com/jerrythomas/exploratory-analysis
'''

'''
modalities

black foreground and white background (16)
purple background and purple foreground (71)
purple foreground and white background (41)
purple foreground and yellow background (8)
white foreground and black background (599)


预处理

后处理

'''
import sys
sys.path.append('../')
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt   
from dataset import 

trains = os.listdir('../data/stage1_train')
tests = os.listdir('../data/stage1_test')


'''
mean pixel
'''

mean_train = []
for im in trains:

