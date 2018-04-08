
# 3D-FCN
INTRA-RETINAL LAYER SEGMENTATION OF OCT USING 3D FULLY CONVOLUTIONAL NETWORKS

This repository contains an implementation of the intra-retinal layer segmentation of optical coherence tomography, using 3D fully convolutional networks introduced by F. Kiaee, H. Rabbani, and H. Fahimi. The original paper can be found at
[https://arxiv.org/…](https://arxiv.org/...)

## Contents

1. A module that constructs the graph of investiated 3D FCN model in tensorflow and performs the training and valiation processes (FCN_scratch_3d.py)
* The function gets the path for reading images (data_dir flag) and also the path for saving ckpt output results (logs_dir flag). If the result path contains previously saved ckpt files, it loads the most recent of them and continues training. otherwise it initializes the weights and starts training.
The mode flag determines the training/test/visualization status.
* The function also gets the model_id arg that helps to check the effect of different terms in cost function and different decoder setting: 

  model_id=0,1,2--> max_unpool decoder setting with 0:standard, 1:weighted, and 2:augmented with Dice score loss function
 
  model_id=3,4,5--> deconvolution decoder setting with 3:standard, 4:weighted, and 5:augmented with Dice score loss function
 
  model_id=6,7,8--> median_unpool decoder setting with 6:standard, 7:weighted, and 8:augmented with Dice score loss function        


2. Utility functions for using with the 3D FCN algorithm, including functions to load biosig and farsiu data. It also contains functions which implement new non-existing tensorflow operations such as median_pool_3d_with_argmedian and unpool_layer_batch_unraveled_indices for 3D unpooling (TensorflowUtils.py)

3. The modules for providing the flow of input images for the graph.

   biosig_data_grndtrth.m: crops and resizes the biosig OCT images to 128x128 and also generates the ground truth segmented images (the original data contains 13 delineated layers from which 8 main layers are used here for the segmentation task)
   
   farsiu_data.m: crops and resizes the Farsiu OCT images to 128x128.
   
   farsiu_grndtrth: generates the ground truth segmented images.
   
   data_cycled_3d_to_binary.py: generates .bin files which for each OCT image contains three heightxwidthxdepth bytes corresponding to the image, its grand truth segmentation and loss function weights. The height and width are 128. In order to make sure that the 3d input volume cycles through the B-scans, if the number of B-scans in OCT volume of a subject is N, it is augmented with the first N-1 B-scans and hence the depth of recorded bytes is 2N-1. Note that for the training process, the desired depth (power of two) of consecutives B-scans is then selected from these 2N-1 B-scans using a random start point.  
 
## Usage

Make sure you're using the tensorflow 0.9.0 version.

1. Download biosig OCT images via [http://www.biosigdata.com/?download=3d-macular-sd-oct-images](http://www.biosigdata.com/?download=3d-macular-sd-oct-images)
Or 
Farsiu OCT images through 
[http://people.duke.edu/~sf59/Chiu_BOE_2014_dataset_htm](http://people.duke.edu/~sf59/Chiu_BOE_2014_dataset_htm)

2. Run farsiu_data.m and farsiu_grandtrth.m to generate .mat files containing cropped and resized images (for Farsiu OCT dataset)
Run biosig_data_grndtrth.m to generate mat files containing cropped and resized images (for Biosig OCT dataset)

3. Run data_cycled_3d_to_binary.py to provide .bin files containing the images, their segmentation and loss weights for both training and test phase. 

4. In order to train the proposed 3D-FCN (with max unpooling decoder), set the mode flag to “train” and execute:
  
   python FCNscratch_3d.py --model_id=3  

5. In order to validate the results set the mode flag to “visualize” and execute:
  
   python FCNscratch_3d.py --model_id=3

