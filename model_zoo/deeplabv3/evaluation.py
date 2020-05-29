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
"""evaluation."""
import argparse
from mindspore import context
from mindspore import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.md_dataset import create_dataset
from src.losses import OhemLoss
from src.miou_precision import MiouPrecision
from src.deeplabv3 import deeplabv3_resnet50
from src.config import config


parser = argparse.ArgumentParser(description="Deeplabv3 evaluation")
parser.add_argument('--epoch_size', type=int, default=2, help='Epoch size.')
parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
parser.add_argument('--batch_size', type=int, default=2, help='Batch size.')
parser.add_argument('--data_url', required=True, default=None, help='Train data url')
parser.add_argument('--checkpoint_url', default=None, help='Checkpoint path')

args_opt = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)
print(args_opt)


if __name__ == "__main__":
    args_opt.crop_size = config.crop_size
    args_opt.base_size = config.crop_size
    eval_dataset = create_dataset(args_opt, args_opt.data_url, args_opt.epoch_size, args_opt.batch_size, usage="eval")
    net = deeplabv3_resnet50(config.seg_num_classes, [args_opt.batch_size, 3, args_opt.crop_size, args_opt.crop_size],
                             infer_scale_sizes=config.eval_scales, atrous_rates=config.atrous_rates,
                             decoder_output_stride=config.decoder_output_stride, output_stride=config.output_stride,
                             fine_tune_batch_norm=config.fine_tune_batch_norm, image_pyramid=config.image_pyramid)
    param_dict = load_checkpoint(args_opt.checkpoint_url)
    load_param_into_net(net, param_dict)
    mIou = MiouPrecision(config.seg_num_classes)
    metrics = {'mIou': mIou}
    loss = OhemLoss(config.seg_num_classes, config.ignore_label)
    model = Model(net, loss, metrics=metrics)
    model.eval(eval_dataset)
