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
"""
eval.
"""
from mindspore import nn
from mindspore.train.model import Model
from mindspore.common import dtype as mstype

from src.dataset import create_dataset
from src.config import set_config
from src.args import eval_parse_args
from src.models import define_net, load_ckpt
from src.utils import switch_precision, set_context

if __name__ == '__main__':
    args_opt = eval_parse_args()
    config = set_config(args_opt)
    set_context(config)
    backbone_net, head_net, net = define_net(config, args_opt.is_training)

    load_ckpt(net, args_opt.pretrain_ckpt)

    switch_precision(net, mstype.float16, config)

    dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=False, config=config)
    step_size = dataset.get_dataset_size()
    if step_size == 0:
        raise ValueError("The step_size of dataset is zero. Check if the images count of eval dataset is more \
            than batch_size in config.py")

    net.set_train(False)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    model = Model(net, loss_fn=loss, metrics={'acc'})

    res = model.eval(dataset)
    print(f"result:{res}\npretrain_ckpt={args_opt.pretrain_ckpt}")
