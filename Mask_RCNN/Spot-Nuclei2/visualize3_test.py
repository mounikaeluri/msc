import os
os.environ['KERAS_BACKEND']='tensorflow'
import sys
sys.path.append('../')

from config2 import *
from dataset import NucleiDataset
import visualize
import numpy as np
import model as modellib
from model import log
import os
import utils
import random
import matplotlib.pyplot as plt
import imageio

if 1:
    plot_dir = 'debug/plot/'
    
    # Training dataset
    dataset_train = NucleiDataset()
    dataset_train.add_nuclei(opt.train_data_root,'train')
    dataset_train.prepare()
    
    # Validation dataset
    dataset_val = NucleiDataset()
    dataset_val.add_nuclei(opt.test_data_root,'test')
    dataset_val.prepare()
    
    
    
    
    class InferenceConfig(Config2):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        
        # Input image resing
        # Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
        # the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
        # be satisfied together the IMAGE_MAX_DIM is enforced.
        IMAGE_MIN_DIM = 512
        IMAGE_MAX_DIM = 512
        # Maximum number of ground truth instances to use in one image
        MAX_GT_INSTANCES = 400
    
    inference_config = InferenceConfig()
    
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=opt.MODEL_DIR)
    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_path = model.find_last()[1]
    
    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
        
if 0:
    # Test on a random image
    for image_id in dataset_val.image_ids:#random.choice(dataset_val.image_ids)
        original_image = dataset_val.load_image(image_id)
        
        fig = plt.figure(figsize=(18, 18))
        plt.imshow(original_image)
        plt.savefig(plot_dir+str(image_id)+'gt'+'.png')
        plt.close(fig)
        
        '''
        visualize val predict
        '''
        results = model.detect([original_image], verbose=1)
        r = results[0]
        visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                        dataset_val.class_names, r['scores'],
                                        figsize=(18, 18), plot_dir=plot_dir, im_id=image_id, alt='p')
        print(r['masks'].shape)

'''
val mAP
0.5 ap 0.9291513532910081
map 0.675
metric 0.59586   (640)
lb 0.409
'''
if 1:
    ths = np.linspace(0.5,0.95,10)
    image_ids = dataset_val.image_ids
    
    APs = []
    for i, image_id in enumerate(image_ids):
        print(i)
        
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP under different iou threshold
        AP = []
        for th in ths:
            AP_th =\
                utils.compute_metric_masks(gt_mask,
                                 r['masks'],
                                 iou_threshold=th)
            AP.append(AP_th)
        APs.append(AP)
        
    print("mAP: ", np.mean(APs))
                            
    
        
            