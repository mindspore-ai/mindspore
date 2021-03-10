"""eval script"""
# Copyright 2021 Huawei Technologies Co., Ltd

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
import numpy as np
from src import ipt
from src.args import args
from src.data.srdata import SRData
from src.metrics import calc_psnr, quantize

from mindspore import context
import mindspore.dataset as de
from mindspore.train.serialization import load_checkpoint, load_param_into_net

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", device_id=0)


def main():
    """eval"""
    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False
    train_dataset = SRData(args, name=args.data_test, train=False, benchmark=False)
    train_de_dataset = de.GeneratorDataset(train_dataset, ['LR', "HR"], shuffle=False)
    train_de_dataset = train_de_dataset.batch(1, drop_remainder=True)
    train_loader = train_de_dataset.create_dict_iterator()

    net_m = ipt.IPT(args)
    print('load mindspore net successfully.')
    if args.pth_path:
        param_dict = load_checkpoint(args.pth_path)
        load_param_into_net(net_m, param_dict)
    net_m.set_train(False)
    num_imgs = train_de_dataset.get_dataset_size()
    psnrs = np.zeros((num_imgs, 1))
    for batch_idx, imgs in enumerate(train_loader):
        lr = imgs['LR']
        hr = imgs['HR']
        hr_np = np.float32(hr.asnumpy())
        pred = net_m.infrc(lr)
        pred_np = np.float32(pred.asnumpy())
        pred_np = quantize(pred_np, 255)
        psnr = calc_psnr(pred_np, hr_np, 4, 255.0, y_only=True)
        psnrs[batch_idx, 0] = psnr
    if args.denoise:
        print('Mean psnr of %s DN_%s is %.4f' % (args.data_test[0], args.sigma, psnrs.mean(axis=0)[0]))
    elif args.derain:
        print('Mean psnr of Derain is %.4f' % (psnrs.mean(axis=0)))
    else:
        print('Mean psnr of %s x%s is %.4f' % (args.data_test[0], args.scale[0], psnrs.mean(axis=0)[0]))


if __name__ == '__main__':
    print("Start main function!")
    main()
