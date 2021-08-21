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

"""eval of DnCNN"""

import os
import ast
import argparse
import glob
import cv2
import numpy as np

import mindspore.common.dtype as mstype
from mindspore.common import set_seed
from mindspore import Tensor, ops, context
from mindspore.train.serialization import load_param_into_net, load_checkpoint

from src.model import DnCNN
from src.config import config
from src.metric import get_PSNR_SSIM
from src.show_image import show_image

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--ckpt_path", type=str, default='', help="restore ckpt file path")
parser.add_argument("--test_data_path", type=str, default='data/Test/Set12',
                    help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
parser.add_argument("--verbose", type=ast.literal_eval, default=False, help='show image result')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
parser.add_argument('--device_id', type=int, default=0, help='Device id')
args = parser.parse_args()

set_seed(1)


def normalize(data):
    return data / 255.


if __name__ == '__main__':
    context.set_context(device_id=args.device_id)
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', save_graphs=False)
    print("======> model test <======\n")
    # load checkpoint file
    dncnn = DnCNN()
    ckpt_start = config.epoch - 4
    base_info = "Dataset: " + args.test_data_path.split('/')[-1] + "    noise level: " + str(args.test_noiseL)
    res_final = 0
    for i in range(5):
        args.ckpt = os.path.join(args.ckpt_path, "dncnn-" + str(ckpt_start + i) + "_1862.ckpt")
        param_dict = load_checkpoint(args.ckpt)
        load_param_into_net(dncnn, param_dict)
        dncnn.set_train(False)
        file_source = glob.glob(os.path.join(args.test_data_path, '*png'))
        file_source.sort()
        psnr_test = 0
        ssim_test = 0
        for f in file_source:
            img = cv2.imread(f)
            img = normalize(np.float32(img[:, :, 0]))
            img = np.expand_dims(img, 0)
            img = np.expand_dims(img, 1)
            source = Tensor(img, dtype=mstype.float32)
            noise = np.random.standard_normal(size=source.shape) * (args.test_noiseL / 255.0)
            noise = Tensor(noise, dtype=mstype.float32)
            noisy_img = source + noise
            out = ops.clip_by_value(noisy_img - dncnn(noisy_img),
                                    Tensor(0., mstype.float32), Tensor(1., mstype.float32))
            if args.verbose:
                save_path = os.path.join('./', args.test_data_path.split('/')[-1] + '_L%s_result' % args.test_noiseL)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                show_image(source, noisy_img, out, os.path.join(save_path, f.split('/')[-1]))
            psnr, ssim = get_PSNR_SSIM(out, source, 1.)
            psnr_test += psnr
            ssim_test += ssim
        psnr_test = psnr_test / len(file_source)
        ssim_test = ssim_test / len(file_source)
        if res_final < psnr_test:
            res_final = psnr_test
            result = base_info + "  PSNR = " + str(psnr_test) + "      ckpt_path: " + args.ckpt
        # result = base_info + "  PSNR = " + str(psnr_test) + "    SSIM = " + str(ssim_test) + "      ckpt_path: " + args.ckpt
        # print(result)
    print(result)
    print("======> Test finish <======\n")
