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
# ============================================================================

import os
import argparse
import logging
from mindspore import context, Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.data_loader import create_dataset, create_cell_nuclei_dataset
from src.unet_medical import UNetMedical
from src.unet_nested import NestedUNet, UNet
from src.config import cfg_unet
from src.utils import UnetEval, TempLoss, dice_coeff

device_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=device_id)

def test_net(data_dir,
             ckpt_path,
             cross_valid_ind=1,
             cfg=None):
    if cfg['model'] == 'unet_medical':
        net = UNetMedical(n_channels=cfg['num_channels'], n_classes=cfg['num_classes'])
    elif cfg['model'] == 'unet_nested':
        net = NestedUNet(in_channel=cfg['num_channels'], n_class=cfg['num_classes'], use_deconv=cfg['use_deconv'],
                         use_bn=cfg['use_bn'], use_ds=False)
    elif cfg['model'] == 'unet_simple':
        net = UNet(in_channel=cfg['num_channels'], n_class=cfg['num_classes'])
    else:
        raise ValueError("Unsupported model: {}".format(cfg['model']))
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    net = UnetEval(net)
    if 'dataset' in cfg and cfg['dataset'] == "Cell_nuclei":
        valid_dataset = create_cell_nuclei_dataset(data_dir, cfg['img_size'], 1, 1, is_train=False,
                                                   eval_resize=cfg["eval_resize"], split=0.8)
    else:
        _, valid_dataset = create_dataset(data_dir, 1, 1, False, cross_valid_ind, False,
                                          do_crop=cfg['crop'], img_size=cfg['img_size'])
    model = Model(net, loss_fn=TempLoss(), metrics={"dice_coeff": dice_coeff(cfg_unet)})

    print("============== Starting Evaluating ============")
    eval_score = model.eval(valid_dataset, dataset_sink_mode=False)["dice_coeff"]
    print("============== Cross valid dice coeff is:", eval_score[0])
    print("============== Cross valid IOU is:", eval_score[1])


def get_args():
    parser = argparse.ArgumentParser(description='Test the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_url', dest='data_url', type=str, default='data/',
                        help='data directory')
    parser.add_argument('-p', '--ckpt_path', dest='ckpt_path', type=str, default='ckpt_unet_medical_adam-1_600.ckpt',
                        help='checkpoint path')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    print("Testing setting:", args)
    test_net(data_dir=args.data_url,
             ckpt_path=args.ckpt_path,
             cross_valid_ind=cfg_unet['cross_valid_ind'],
             cfg=cfg_unet)
