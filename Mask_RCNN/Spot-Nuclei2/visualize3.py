import os
os.environ['KERAS_BACKEND']='tensorflow'
import sys
sys.path.append('../')

from config2 import *
from dataset import NucleiDataset
#import visualize
import numpy as np
import model as modellib
from model import log
import os
import utils
import random
#import matplotlib.pyplot as plt
import imageio
import scipy

from skimage import morphology
from skimage.morphology import binary_closing, binary_opening, disk, binary_dilation

def dilation(mask):
    return binary_dilation(mask, disk(1))


if 1:
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
        
if 0:
    # Test on a random image
    for image_id in dataset_val.image_ids:#random.choice(dataset_val.image_ids)
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
    #    plt.figure(figsize=(18, 18))
    #    plt.imshow(original_mask)#, cmap='gray')
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

'''
val mAP
0.5 ap 0.9291513532910081
map 0.675
metric 0.59586  
lb 0.409

512 metric:  0.5890161
lb 0.407
to 256: bilinear: 0.589866  nearest: 0.5564196
to 1024: bilinear:0.589016   nearest: 0.589016
dilation 1: 0.5719273
clean: 0.528974685628

膨胀后与语义分割取交集，再膨胀

'''
if 1:
    ths = np.linspace(0.5,0.95,10)
    image_ids = dataset_val.image_ids
    
    APs = []
    for i, image_id in enumerate(image_ids):
        #print(i)
        
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        
        # test, bilinear:   nearest:
        #for j in range(r['masks'].shape[-1]-1):
        #    r['masks'][:,:,j]= clean_img(r['masks'][:,:,j])
        #    mini = scipy.misc.imresize(r['masks'][:,:,i], (1024, 1024), interp='bilinear')
        #    r['masks'][:,:,i] =  scipy.misc.imresize(mini, (512, 512), interp='bilinear')
            #print(r['masks'][:,:,i].max())
        #    r['masks'][:,:,i] = np.where(r['masks'][:,:,i]>=0.5,1,0).astype(np.uint8)
        
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
                            
    
        
            
