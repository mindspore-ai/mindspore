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
import datetime
import argparse
import os
import time
import glob
import pandas as pd
import numpy as np
import PIL.Image as Image

import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore.common import set_seed
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore import load_checkpoint, load_param_into_net

from src.logger import get_logger
from src.models import BRDNet

## Params
parser = argparse.ArgumentParser()

parser.add_argument('--test_dir', default='./Test/Kodak24/'
                    , type=str, help='directory of test dataset')
parser.add_argument('--sigma', default=15, type=int, help='noise level')
parser.add_argument('--channel', default=3, type=int
                    , help='image channel, 3 for color, 1 for gray')
parser.add_argument('--pretrain_path', default=None, type=str, help='path of pre-trained model')
parser.add_argument('--ckpt_name', default=None, type=str, help='ckpt_name')
parser.add_argument('--use_modelarts', type=int, default=0
                    , help='1 for True, 0 for False;when set True, we should load dataset from obs with moxing')
parser.add_argument('--train_url', type=str, default='train_url/'
                    , help='needed by modelarts, but we donot use it because the name is ambiguous')
parser.add_argument('--data_url', type=str, default='data_url/'
                    , help='needed by modelarts, but we donot use it because the name is ambiguous')
parser.add_argument('--output_path', type=str, default='./output/'
                    , help='output_path,when use_modelarts is set True, it will be cache/output/')
parser.add_argument('--outer_path', type=str, default='s3://output/'
                    , help='obs path,to store e.g ckpt files ')

parser.add_argument('--device_target', type=str, default='Ascend',
                    help='device where the code will be implemented. (Default: Ascend)')

set_seed(1)

args = parser.parse_args()
save_dir = os.path.join(args.output_path, 'sigma_' + str(args.sigma) + \
                        '_' + datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))

if not args.use_modelarts and not os.path.exists(save_dir):
    os.makedirs(save_dir)

def test(model_path):
    args.logger.info('Start to test on {}'.format(args.test_dir))
    out_dir = os.path.join(save_dir, args.test_dir.split('/')[-2]) # args.test_dir must end by '/'
    if not args.use_modelarts and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model = BRDNet(args.channel)
    args.logger.info('load test weights from '+str(model_path))
    load_param_into_net(model, load_checkpoint(model_path))
    name = []
    psnr = []   #after denoise
    ssim = []   #after denoise
    psnr_b = [] #before denoise
    ssim_b = [] #before denoise

    if args.use_modelarts:
        args.logger.info("copying test dataset from obs to cache....")
        mox.file.copy_parallel(args.test_dir, 'cache/test')
        args.logger.info("copying test dataset finished....")
        args.test_dir = 'cache/test/'

    file_list = glob.glob(args.test_dir+'*') # args.test_dir must end by '/'
    model.set_train(False)

    cast = P.Cast()
    transpose = P.Transpose()
    expand_dims = P.ExpandDims()
    compare_psnr = nn.PSNR()
    compare_ssim = nn.SSIM()

    args.logger.info("start testing....")
    start_time = time.time()
    for file in file_list:
        suffix = file.split('.')[-1]
        # read image
        if args.channel == 3:
            img_clean = np.array(Image.open(file), dtype='float32') / 255.0
        else:
            img_clean = np.expand_dims(np.array(Image.open(file).convert('L'), dtype='float32') / 255.0, axis=2)

        np.random.seed(0) #obtain the same random data when it is in the test phase
        img_test = img_clean + np.random.normal(0, args.sigma/255.0, img_clean.shape)

        img_clean = Tensor(img_clean, mindspore.float32) #HWC
        img_test = Tensor(img_test, mindspore.float32)   #HWC

        # predict
        img_clean = expand_dims(transpose(img_clean, (2, 0, 1)), 0)#NCHW
        img_test = expand_dims(transpose(img_test, (2, 0, 1)), 0)#NCHW

        y_predict = model(img_test)    #NCHW

        # calculate numeric metrics

        img_out = C.clip_by_value(y_predict, 0, 1)

        psnr_noise, psnr_denoised = compare_psnr(img_clean, img_test), compare_psnr(img_clean, img_out)
        ssim_noise, ssim_denoised = compare_ssim(img_clean, img_test), compare_ssim(img_clean, img_out)


        psnr.append(psnr_denoised.asnumpy()[0])
        ssim.append(ssim_denoised.asnumpy()[0])
        psnr_b.append(psnr_noise.asnumpy()[0])
        ssim_b.append(ssim_noise.asnumpy()[0])

        # save images
        filename = file.split('/')[-1].split('.')[0]    # get the name of image file
        name.append(filename)

        if not args.use_modelarts:
            # inner the operation 'Image.save', it will first check the file \
            # existence of same name, that is not allowed on modelarts
            img_test = cast(img_test*255, mindspore.uint8).asnumpy()
            img_test = img_test.squeeze(0).transpose((1, 2, 0)) #turn into HWC to save as an image
            img_test = Image.fromarray(img_test)
            img_test.save(os.path.join(out_dir, filename+'_sigma'+'{}_psnr{:.2f}.'\
                            .format(args.sigma, psnr_noise.asnumpy()[0])+str(suffix)))
            img_out = cast(img_out*255, mindspore.uint8).asnumpy()
            img_out = img_out.squeeze(0).transpose((1, 2, 0)) #turn into HWC to save as an image
            img_out = Image.fromarray(img_out)
            img_out.save(os.path.join(out_dir, filename+'_psnr{:.2f}.'.format(psnr_denoised.asnumpy()[0])+str(suffix)))


    psnr_avg = sum(psnr)/len(psnr)
    ssim_avg = sum(ssim)/len(ssim)
    psnr_avg_b = sum(psnr_b)/len(psnr_b)
    ssim_avg_b = sum(ssim_b)/len(ssim_b)
    name.append('Average')
    psnr.append(psnr_avg)
    ssim.append(ssim_avg)
    psnr_b.append(psnr_avg_b)
    ssim_b.append(ssim_avg_b)
    args.logger.info('Before denoise: Average PSNR_b = {0:.2f}, \
                     SSIM_b = {1:.2f};After denoise: Average PSNR = {2:.2f}, SSIM = {3:.2f}'\
                     .format(psnr_avg_b, ssim_avg_b, psnr_avg, ssim_avg))
    args.logger.info("testing finished....")
    time_used = time.time() - start_time
    args.logger.info("time cost:"+str(time_used)+" seconds!")
    if not args.use_modelarts:
        pd.DataFrame({'name': np.array(name), 'psnr_b': np.array(psnr_b), \
                      'psnr': np.array(psnr), 'ssim_b': np.array(ssim_b), \
                      'ssim': np.array(ssim)}).to_csv(out_dir+'/metrics.csv', index=True)

if __name__ == '__main__':

    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True,
                        device_target=args.device_target, device_id=device_id, save_graphs=False)

    args.logger = get_logger(save_dir, "BRDNet", 0)
    args.logger.save_args(args)

    if args.use_modelarts:
        import moxing as mox
        args.logger.info("copying test weights from obs to cache....")
        mox.file.copy_parallel(args.pretrain_path, 'cache/weight')
        args.logger.info("copying test weights finished....")
        args.pretrain_path = 'cache/weight/'

    test(os.path.join(args.pretrain_path, args.ckpt_name))

    if args.use_modelarts:
        args.logger.info("copying files from cache to obs....")
        mox.file.copy_parallel(save_dir, args.outer_path)
        args.logger.info("copying finished....")
