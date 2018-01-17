# Kaggle-CMI
Repository for Kaggle's competition "IEEE's Signal Processing Society - Camera Model Identification".

## 1. Data

### Train
* Images in the training set were captured with 10 different camera models, a single device per model, with 275 full images from each device. 
* The pixels of each image are different for each device (same device, same pixels).

### Test Images 
* In the test set were captured with the same 10 camera models, but using a second device. 
* While the train data includes full images, the test data contains only single 512 x 512 pixel blocks cropped from the center of a single image taken with the device. 
* No two image blocks come from the same original image.
* Half of the images in the test set have been altered. The image names indicate whether or not they were manipulated (\_manip) from the original or unaltered (\_unalt). 
 * Operations 
     * JPEG compression with quality factor = 70
     * JPEG compression with quality factor = 90
     * resizing (via bicubic interpolation) by a factor of 0.5
     * resizing (via bicubic interpolation) by a factor of 0.8
     * resizing (via bicubic interpolation) by a factor of 1.5
     * resizing (via bicubic interpolation) by a factor of 2.0
     * gamma correction using gamma = 0.8
     * gamma correction using gamma = 1.2

### Pre-processing
