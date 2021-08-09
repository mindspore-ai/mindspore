# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================
""" generate image from 310 out """
import os
import glob
import numpy as np
from PIL import Image

file_dir = './scripts/result_Files/'
file_list = glob.glob(file_dir + '*.bin')
rgb_path = './picture/'

if __name__ == '__main__':
    for file_path in file_list:
        file_name = os.path.basename(file_path)[0: 6]
        output = np.fromfile(file_path, dtype=np.float32)
        output = output.reshape((1, 3, 2848, 4256))
        output = np.minimum(np.maximum(output, 0), 1)
        output = np.trunc(output[0] * 255)
        output = output.astype(np.int8)
        output = output.transpose([1, 2, 0])
        im = Image.fromarray(output, 'RGB')
        im.save(rgb_path + file_name + '.png')
