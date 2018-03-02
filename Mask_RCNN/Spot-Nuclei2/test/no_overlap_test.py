import numpy as np

predicts = np.array([[[1,0],[1,0]], [[0,0],[1,0]]])
sum_predicts = np.sum(predicts, axis=2)
sum_predicts[sum_predicts>=2] = 0

predicts = predicts * sum_predicts



scores = np.array([1,2])
predicts = np.array([[[1,0],[1,0]], [[0,0],[1,1]]])
print(predicts[:,:,0])
print(predicts[:,:,1])
sum_predicts = np.sum(predicts, axis=2)
bool_mask = sum_predicts>=2
print(bool_mask)

"""
[[1 1]
 [0 1]]
[[0 0]
 [0 1]]
[[False False]
 [False  True]]
"""

