# -*- coding: utf-8 -*-
"""
测试resize小，再resize回来，mask是否有偏差！
"""

'''
(Training examples, Test examples): ( 670 , 65 )
(256, 256, 3) : 334
(1024, 1024, 3) : 16
(520, 696, 3) : 92
(360, 360, 3) : 91
(512, 640, 3) : 13
(256, 320, 3) : 112
(1040, 1388, 3) : 1
(260, 347, 3) : 5
(603, 1272, 3) : 6
'''

import numpy as np
import scipy
import skimage

#h = 56
#w = 511

h = 512
w = 512

gt = np.random.randint(2, size=(h, w))*255
gt.astype(float)
mini = scipy.misc.imresize(gt, (512, 512), interp='nearest')
pr =  scipy.misc.imresize(mini, (h, w), interp='nearest')
print(np.sum(pr-gt)/255/h/w)


mini = scipy.misc.imresize(gt, (512, 512))
pr =  scipy.misc.imresize(mini, (h, w))
print(np.sum(pr-gt)/255/h/w)


mini = scipy.misc.imresize(gt, (512, 512), interp='bilinear')
pr =  scipy.misc.imresize(mini, (h, w), interp='bilinear')
print(np.sum(pr-gt)/255/h/w)



#mini = skimage.transform.resize(gt, (512, 512))
#pr =  skimage.transform.resize(mini, (h, w))
#print(np.sum(pr-gt)/np.sum(gt))
