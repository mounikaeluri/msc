import os
os.environ['KERAS_BACKEND']='tensorflow'
import sys
sys.path.append('../')

from config2 import *
from dataset import NucleiDataset
import numpy as np
import model as modellib
from model import log
import utils
import random

# Training dataset
dataset_train = NucleiDataset()
dataset_train.add_nuclei(opt.train_data_root,'train')
dataset_train.prepare()

# Validation dataset
dataset_val = NucleiDataset()
dataset_val.add_nuclei(opt.val_data_root,'val')
dataset_val.prepare()

"""
# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
 """   

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=opt,
                          model_dir=opt.MODEL_DIR)


# Which weights to start with?
init_with = opt.init_with  # imagenet, coco, or last
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    if not os.path.exists(opt.COCO_MODEL_PATH):
        utils.download_trained_weights(opt.COCO_MODEL_PATH)
    
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(opt.COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)
    

'''
train

1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers 
    (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, 
    pass layers='heads' to the train() function.
2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. 
    Simply pass layers="all to train all layers.
'''


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
"""
model.train(dataset_train, dataset_val, 
            learning_rate=opt.LEARNING_RATE, 
            epochs=10, 
            layers='heads')
"""

model.train(dataset_train, dataset_val,
            learning_rate=opt.LEARNING_RATE,
            epochs=40,
            layers='all')



## Fine tune all layers
## Passing layers="all" trains all layers. You can also 
## pass a regular expression to select which layers to
## train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=opt.LEARNING_RATE/10,
            epochs=80, 
            layers="all")

model.train(dataset_train, dataset_val,
            learning_rate=opt.LEARNING_RATE/100,
            epochs=120,
            layers='all')

