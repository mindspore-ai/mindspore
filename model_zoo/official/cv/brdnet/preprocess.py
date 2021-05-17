# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import argparse
import os
import glob
import numpy as np
import PIL.Image as Image

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, required=True,
                    help='directory to store the image with noise')
parser.add_argument('--image_path', type=str, required=True,
                    help='directory of image to add noise')
parser.add_argument('--channel', type=int, default=3
                    , help='image channel, 3 for color, 1 for gray')
parser.add_argument('--sigma', type=int, default=15, help='level of noise')
args = parser.parse_args()

def add_noise(out_dir, image_path, channel, sigma):
    file_list = glob.glob(image_path+'*') # image_path must end by '/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for file in file_list:
        print("Adding noise to: ", file)
        # read image
        if channel == 3:
            img_clean = np.array(Image.open(file), dtype='float32') / 255.0
        else:
            img_clean = np.expand_dims(np.array(Image.open(file).convert('L'), dtype='float32') / 255.0, axis=2)

        np.random.seed(0) #obtain the same random data when it is in the test phase
        img_test = img_clean + np.random.normal(0, sigma/255.0, img_clean.shape).astype(np.float32)#HWC
        img_test = np.expand_dims(img_test.transpose((2, 0, 1)), 0)#NCHW
        #img_test = np.clip(img_test, 0, 1)

        filename = file.split('/')[-1].split('.')[0]    # get the name of image file
        img_test.tofile(os.path.join(out_dir, filename+'_noise.bin'))

if __name__ == "__main__":
    add_noise(args.out_dir, args.image_path, args.channel, args.sigma)
