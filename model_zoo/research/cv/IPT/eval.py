"""eval script"""
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

import os
import numpy as np
import mindspore.dataset as ds
from mindspore import Tensor, context
from mindspore.common import dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.args import args
import src.ipt_model as ipt
from src.data.srdata import SRData
from src.metrics import calc_psnr, quantize

device_id = int(os.getenv('DEVICE_ID', '0'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id, save_graphs=False)
context.set_context(max_call_depth=10000)

def sub_mean(x):
    red_channel_mean = 0.4488 * 255
    green_channel_mean = 0.4371 * 255
    blue_channel_mean = 0.4040 * 255
    x[:, 0, :, :] -= red_channel_mean
    x[:, 1, :, :] -= green_channel_mean
    x[:, 2, :, :] -= blue_channel_mean
    return x

def add_mean(x):
    red_channel_mean = 0.4488 * 255
    green_channel_mean = 0.4371 * 255
    blue_channel_mean = 0.4040 * 255
    x[:, 0, :, :] += red_channel_mean
    x[:, 1, :, :] += green_channel_mean
    x[:, 2, :, :] += blue_channel_mean
    return x

def eval_net():
    """eval"""
    args.batch_size = 128
    args.decay = 70
    args.patch_size = 48
    args.num_queries = 6
    args.model = 'vtip'
    args.num_layers = 4

    if args.epochs == 0:
        args.epochs = 1e8

    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False
    train_dataset = SRData(args, name=args.data_test, train=False, benchmark=False)
    train_de_dataset = ds.GeneratorDataset(train_dataset, ['LR', 'HR', "idx", "filename"], shuffle=False)
    train_de_dataset = train_de_dataset.batch(1, drop_remainder=True)
    train_loader = train_de_dataset.create_dict_iterator(output_numpy=True)

    net_m = ipt.IPT(args)
    if args.pth_path:
        param_dict = load_checkpoint(args.pth_path)
        load_param_into_net(net_m, param_dict)
    net_m.set_train(False)
    idx = Tensor(np.ones(args.task_id), mstype.int32)
    inference = ipt.IPT_post(net_m, args)
    print('load mindspore net successfully.')
    num_imgs = train_de_dataset.get_dataset_size()
    psnrs = np.zeros((num_imgs, 1))
    for batch_idx, imgs in enumerate(train_loader):
        lr = imgs['LR']
        hr = imgs['HR']
        lr = sub_mean(lr)
        lr = Tensor(lr, mstype.float32)
        pred = inference.forward(lr, idx)
        pred_np = add_mean(pred.asnumpy())
        pred_np = quantize(pred_np, 255)
        psnr = calc_psnr(pred_np, hr, 4, 255.0)
        print("current psnr: ", psnr)
        psnrs[batch_idx, 0] = psnr
    if args.denoise:
        print('Mean psnr of %s DN_%s is %.4f' % (args.data_test[0], args.sigma, psnrs.mean(axis=0)[0]))
    elif args.derain:
        print('Mean psnr of Derain is %.4f' % (psnrs.mean(axis=0)))
    else:
        print('Mean psnr of %s x%s is %.4f' % (args.data_test[0], args.scale[0], psnrs.mean(axis=0)[0]))

if __name__ == '__main__':
    print("Start eval function!")
    eval_net()
