# Copyright 2025 Huawei Technologies Co., Ltd
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
"""resnet train case."""
import mindspore as ms
from mindspore import nn, Callback
from tests.st.networks.models.resnet50.src.config import config
from tests.st.networks.models.resnet50.src.resnet import resnet50
from tests.st.networks.models.resnet50.src.metric import DistAccuracy, ClassifyCorrectCell
from tests.st.networks.models.resnet50.src.dataset import create_dataset
from tests.st.networks.models.resnet50.src.lr_generator import get_learning_rate
from tests.st.networks.models.resnet50.src.CrossEntropySmooth import CrossEntropySmooth

TRAIN_PATH = "/home/workspace/mindspore_dataset/imagenet/imagenet_original/train"
EVAL_PATH = "/home/workspace/mindspore_dataset/imagenet/imagenet_original/val"
ms.set_seed(1)

class StopEarly(Callback):
    """Stop training early callback."""
    def on_train_step_end(self, run_context):
        run_context.request_stop()

def get_optimizer(net, step_size):
    # optimizer
    lr = ms.Tensor(get_learning_rate(lr_init=config.lr_init, lr_end=0.0, lr_max=config.lr_max,
                                     warmup_epochs=config.warmup_epochs, total_epochs=config.epoch_size,
                                     steps_per_epoch=step_size, lr_decay_mode=config.lr_decay_mode))
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params, 'weight_decay': 0.0},
                    {'order_params': net.trainable_params()}]

    opt = nn.Momentum(group_params, lr, config.momentum,
                        loss_scale=config.loss_scale, use_nesterov=config.use_nesterov)
    return opt


def run_train():
    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
    ms.set_deterministic(True)
    ms.set_seed(1)
    ms.communication.init()
    ms.context.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                         gradients_mean=True, all_reduce_fusion_config=[107, 160])
    net = resnet50(class_num=config.class_num)
    dist_eval_network = ClassifyCorrectCell(net)

    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropySmooth(sparse=True, reduction="mean",
                              smooth_factor=config.label_smooth_factor, num_classes=config.class_num)

    # dataset
    dataset = create_dataset(dataset_path=TRAIN_PATH, do_train=True, repeat_num=1, batch_size=config.batch_size)
    step_size = dataset.get_dataset_size()

    loss_scale = ms.FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    opt = get_optimizer(net, step_size)

    model = ms.Model(net, loss_fn=loss, optimizer=opt,
                     loss_scale_manager=loss_scale, amp_level="O2", keep_batchnorm_fp32=False,
                     metrics={'acc': DistAccuracy(batch_size=config.eval_batch_size,
                                                  device_num=ms.communication.get_group_size())},
                     eval_network=dist_eval_network)
    stop_early = StopEarly()
    model.train(1, dataset, callbacks=stop_early, dataset_sink_mode=False)

if __name__ == '__main__':
    run_train()
