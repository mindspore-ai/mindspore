# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from mindspore import log as logger
import mindspore.dataset.engine as de
import mindspore.dataset.transforms.vision.py_transforms as F

DATA_DIR = "../data/dataset/testImageNetData/train/"


def visualize(image_original, image_auto_contrast):
    """
    visualizes the image using DE op and Numpy op
    """
    num = len(image_auto_contrast)
    for i in range(num):
        plt.subplot(2, num, i + 1)
        plt.imshow(image_original[i])
        plt.title("Original image")

        plt.subplot(2, num, i + num + 1)
        plt.imshow(image_auto_contrast[i])
        plt.title("DE AutoContrast image")

    plt.show()
    

def test_auto_contrast(plot=False):
    """
    Test AutoContrast
    """
    logger.info("Test AutoContrast")
    
    # Original Images
    ds = de.ImageFolderDatasetV2(dataset_dir=DATA_DIR, shuffle=False)    
    
    transforms_original = F.ComposeOp([F.Decode(),
                                       F.Resize((224,224)),
                                       F.ToTensor()])    
    
    ds_original = ds.map(input_columns="image",
                         operations=transforms_original())
    
    ds_original = ds_original.batch(512)
            
    for idx, (image,label) in enumerate(ds_original):
        if idx == 0:
            images_original = np.transpose(image, (0, 2,3,1))
        else:
            images_original = np.append(images_original,
                                        np.transpose(image, (0, 2,3,1)),
                                        axis=0)    

    # AutoContrast Images
    ds = de.ImageFolderDatasetV2(dataset_dir=DATA_DIR, shuffle=False)    
    
    transforms_auto_contrast = F.ComposeOp([F.Decode(),
                                            F.Resize((224,224)),
                                            F.AutoContrast(),
                                            F.ToTensor()])    
    
    ds_auto_contrast = ds.map(input_columns="image",
                                 operations=transforms_auto_contrast())
    
    ds_auto_contrast = ds_auto_contrast.batch(512)    
      
    for idx, (image,label) in enumerate(ds_auto_contrast):
        if idx == 0:
            images_auto_contrast = np.transpose(image, (0, 2,3,1))
        else:
            images_auto_contrast = np.append(images_auto_contrast,
                                      np.transpose(image, (0, 2,3,1)),
                                      axis=0)
    
    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = np.mean((images_auto_contrast[i]-images_original[i])**2)
    logger.info("MSE= {}".format(str(np.mean(mse))))
    
    if plot:
        visualize(images_original, images_auto_contrast)
        

if __name__ == "__main__":
    test_auto_contrast(plot=True)
    
