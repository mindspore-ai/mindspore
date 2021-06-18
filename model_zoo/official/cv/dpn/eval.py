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
"""DPN model eval with MindSpore"""
from mindspore import context
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.imagenet_dataset import classification_dataset
from src.dpn import dpns
from src.crossentropy import CrossEntropy
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id


set_seed(1)


# set context
context.set_context(mode=context.GRAPH_MODE,
                    device_target=config.device_target, save_graphs=False, device_id=get_device_id())


@moxing_wrapper(pre_process=None)
def dpn_evaluate():
    # create evaluate dataset
    # eval_path = os.path.join(config.eval_data_dir, 'val')
    eval_dataset = classification_dataset(config.eval_data_dir,
                                          image_size=config.image_size,
                                          num_parallel_workers=config.num_parallel_workers,
                                          per_batch_size=config.batch_size,
                                          max_epoch=1,
                                          rank=config.rank,
                                          shuffle=False,
                                          group_size=config.group_size,
                                          mode='eval')

    # create network
    net = dpns[config.backbone](num_classes=config.num_classes)
    # load checkpoint
    load_param_into_net(net, load_checkpoint(config.checkpoint_path))
    print("load checkpoint from [{}].".format(config.checkpoint_path))
    # loss
    if config.dataset == "imagenet-1K":
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    else:
        if not config.label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.num_classes)

        # create model
    model = Model(net, keep_batchnorm_fp32=False, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})
    # evaluate
    output = model.eval(eval_dataset)
    print(f'Evaluation result: {output}.')


if __name__ == '__main__':
    dpn_evaluate()
    print('DPN evaluate success!')
