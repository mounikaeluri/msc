import os
os.environ['KERAS_BACKEND']='tensorflow'
import sys
sys.path.append('../')

from config2 import *
from dataset import NucleiDataset
import numpy as np
import model as modellib
import functions as f

from skimage import morphology
from skimage.morphology import binary_closing, binary_opening, disk, binary_dilation

def dilation(mask, q):
    return binary_dilation(mask, disk(q))


dataset = NucleiDataset()
dataset.add_nuclei('../../Spot-Nuclei-master/data/stage1_test/','test')
dataset.prepare()



class InferenceConfig(Config2):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    MEAN_PIXEL = np.array([56.02288505, 54.02376286, 54.26675248])

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



'''
load test dataset one by one. Note that masks are resized (back) in model.detect
rle2csv
'''        
ImageId = []
EncodedPixels = []
    
for i, image_id in enumerate(dataset.image_ids):
    print(i)
    
    # Load image
    image = dataset.load_image(image_id)

    # Run detection
    r = model.detect([image], verbose=0)[0]
    
    # post
    semantic = dataset.load_semantic(image_id)
    for j in range(r['masks'].shape[-1]-1):
        r['masks'][:,:,j]= dilation(r['masks'][:,:,j], 2)
        r['masks'][:,:,j] = r['masks'][:,:,j] * semantic
            
    masks = r['masks'] #[H, W, N] instance binary masks
    
    img_name = dataset.image_info[image_id]['name']
    
    
    ImageId_batch, EncodedPixels_batch = f.numpy2encoding_no_overlap2(masks, img_name, r['scores'])
    ImageId += ImageId_batch
    EncodedPixels += EncodedPixels_batch

f.write2csv('results/'+'MaskRCNN'+'_test.csv', ImageId, EncodedPixels)
print(len(np.unique(ImageId))) 
    
    
