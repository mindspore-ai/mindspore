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
"""train."""
import argparse
import time
from mindspore import context
from mindspore.nn.optim.momentum import Momentum
from mindspore import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import Callback
from src.md_dataset import create_dataset
from src.losses import OhemLoss
from src.deeplabv3 import deeplabv3_resnet50
from src.config import config
parser = argparse.ArgumentParser(description="Deeplabv3 training")
parser.add_argument('--data_url', required=True, default=None, help='Train data url')
parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
parser.add_argument('--checkpoint_url', default=None, help='Checkpoint path')
args_opt = parser.parse_args()
print(args_opt)
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)

class LossCallBack(Callback):
    """
    Monitor the loss in training.
    Note:
        if per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """
    def __init__(self, data_size, per_print_times=1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0")
        self.data_size = data_size
        self._per_print_times = per_print_times
        self.time = 1000
        self.loss = 0
    def epoch_begin(self, run_context):
        self.epoch_time = time.time()
    def step_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        self.time = epoch_mseconds / self.data_size
        self.loss += cb_params.net_outputs
        print("epoch: {}, step: {}, outputs are {}".format(cb_params.cur_epoch_num, cb_params.cur_step_num,
                                                           str(cb_params.net_outputs)))

def model_fine_tune(flags, train_net, fix_weight_layer):
    checkpoint_path = flags.checkpoint_url
    if checkpoint_path is None:
        return
    param_dict = load_checkpoint(checkpoint_path)
    load_param_into_net(train_net, param_dict)
    for para in train_net.trainable_params():
        if fix_weight_layer in para.name:
            para.requires_grad = False

if __name__ == "__main__":
    start_time = time.time()
    epoch_size = 3
    args_opt.base_size = config.crop_size
    args_opt.crop_size = config.crop_size
    train_dataset = create_dataset(args_opt, args_opt.data_url, 1, config.batch_size,
                                   usage="train", shuffle=False)
    dataset_size = train_dataset.get_dataset_size()
    callback = LossCallBack(dataset_size)
    net = deeplabv3_resnet50(config.seg_num_classes, [config.batch_size, 3, args_opt.crop_size, args_opt.crop_size],
                             infer_scale_sizes=config.eval_scales, atrous_rates=config.atrous_rates,
                             decoder_output_stride=config.decoder_output_stride, output_stride=config.output_stride,
                             fine_tune_batch_norm=config.fine_tune_batch_norm, image_pyramid=config.image_pyramid)
    net.set_train()
    model_fine_tune(args_opt, net, 'layer')
    loss = OhemLoss(config.seg_num_classes, config.ignore_label)
    opt = Momentum(filter(lambda x: 'beta' not in x.name and 'gamma' not in x.name and 'depth' not in x.name and 'bias' not in x.name, net.trainable_params()), learning_rate=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    model = Model(net, loss, opt)
    model.train(epoch_size, train_dataset, callback)
    print(time.time() - start_time)
    print("expect loss: ", callback.loss / 3)
    print("expect time: ", callback.time)
    expect_loss = 0.5
    expect_time = 35
    assert callback.loss.asnumpy() / 3 <= expect_loss
    assert callback.time <= expect_time
