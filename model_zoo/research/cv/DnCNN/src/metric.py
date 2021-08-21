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

"""metric of DnCNN"""

import numpy as np
import mindspore as ms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def get_PSNR_SSIM(original_clean_image, result_image, data_range):
    """compute the PSNR and SSIM between the original image and the result image(get by learning)"""
    # convert to numpy array
    if not isinstance(original_clean_image, np.ndarray):
        if isinstance(original_clean_image, ms.Tensor):
            original_clean_image = original_clean_image.asnumpy()
    if not isinstance(result_image, np.ndarray):
        if isinstance(result_image, ms.Tensor):
            result_image = result_image.asnumpy()
    image_num = original_clean_image.shape[0]
    PSNR = []
    SSIM = []
    for i in range(image_num):
        PSNR.append(peak_signal_noise_ratio(original_clean_image[i, :, :, :],
                                            result_image[i, :, :, :],
                                            data_range=data_range))
        orginal_clean_image_ = np.expand_dims(np.squeeze(original_clean_image), axis=-1)
        result_image_ = np.expand_dims(np.squeeze(result_image), axis=-1)
        SSIM.append(structural_similarity(np.array(orginal_clean_image_, dtype=np.float16),
                                          np.array(result_image_, dtype=np.float16),
                                          data_range=data_range, multichannel=True))
    return np.mean(PSNR), np.mean(SSIM)
