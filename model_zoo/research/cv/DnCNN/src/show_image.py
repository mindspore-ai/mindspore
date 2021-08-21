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

"""show the result image"""

import numpy as np
from PIL import Image

def show_image(original_image, noisy_image, result_image, save_path):
    """
     show the original image,
     the noisy image
     and the result image(get by learning) during test
    """
    original_image = ((np.squeeze(original_image.asnumpy()) + 1) / 2.0 * 255.0).astype(np.uint8)
    noisy_image = ((np.squeeze(noisy_image.asnumpy()) + 1) / 2.0 * 255.0).astype(np.uint8)
    result_image = ((np.squeeze(result_image.asnumpy()) + 1) / 2.0 * 255.0).astype(np.uint8)
    image_s_numpy = np.concatenate((original_image, noisy_image, result_image))
    image_pil = Image.fromarray(image_s_numpy, mode='L')
    image_pil.save(save_path)
