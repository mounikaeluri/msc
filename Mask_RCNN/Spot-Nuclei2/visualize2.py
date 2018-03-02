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

if 0:
    plot_dir = 'debug/plot/'
    
    # Training dataset
    dataset_train = NucleiDataset()
    dataset_train.add_nuclei(opt.train_data_root,'train')
    dataset_train.prepare()
    
    # Validation dataset
    dataset_val = NucleiDataset()
    dataset_val.add_nuclei(opt.val_data_root,'val')
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
        
    
# Test on a random image
for image_id in [64]:#dataset_val.image_ids:#random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask, original_size =\
        modellib.load_image_gt2(dataset_val, inference_config, 
                               image_id, use_mini_mask=False)
    
    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)
    
    '''
    visualize val gt 
    '''
#    plt.savefig(str(image_id)+'s.png')
    
    #resize and resize back
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                dataset_train.class_names,
                                figsize=(18, 18), plot_dir=plot_dir, im_id=image_id, alt='gt')
    print(original_size.shape)
    print(original_image.shape)
    print(gt_mask.shape)#(256, 256, 22)
    
    #display original masks in one figure wihout resize
    original_mask = imageio.imread(
            dataset_val.image_info[image_id]['mask_dir'].replace('masks/','images/') + 'mask.png')
    plt.figure(figsize=(18, 18))
    plt.imshow(original_mask)#, cmap='gray')
    print(original_mask.shape)#(360, 360)
    
    '''
    visualize val predict
    '''
    results = model.detect([original_image], verbose=1)
    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                    dataset_val.class_names, r['scores'],
                                    figsize=(18, 18), plot_dir=plot_dir, im_id=image_id, alt='p')
    print(r['masks'].shape)
    print(r['scores'])

'''
val mAP 
'''
if 0:
    ths = np.linspace(0.5,0.95,10)
    image_ids = dataset_val.image_ids
    
    APs = []
    for image_id in image_ids:
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
            AP_th, precisions, recalls, overlaps =\
                utils.compute_ap(gt_bbox, gt_class_id,
                                 r["rois"], r["class_ids"], r["scores"],
                                 iou_threshold=th)
            AP.append(AP_th)
        APs.append(AP)
        
    print("mAP: ", np.mean(APs))
                            
    
        
            