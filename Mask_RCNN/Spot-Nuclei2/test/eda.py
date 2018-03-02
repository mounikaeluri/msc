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

'''

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt   

df = pd.read_csv('../debug/classes.csv')
df[['name','_']] = df['filename'].str.split('.', expand = True)
df.drop('_',1,inplace=True)

trains = os.listdir('../data/stage1_train')
tests = os.listdir('../data/stage1_test')

if 0:
    def name2splits(name):
        if name in trains:
            return '1'
        return '0'
    #df.to_csv('../debug/class_split.csv')
    #
    df['splits'] = np.vectorize(name2splits)(df['name'])
    
    df['splits'].value_counts().plot.bar()  
    plt.show() 
    
    df['complex'] = df[['foreground', 'background']].apply(lambda x: '_'.join(x), axis=1)
    fig = plt.figure(figsize=(18, 18))
    df.hist(column='complex',by=df['splits'],sharex=True)
    plt.xticks(fontsize = 20) 
    plt.show()
    plt.close(fig)


'''
mean pixel
'''

