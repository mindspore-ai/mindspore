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
"""eval script"""

import os
import numpy as np

from mindspore import context
from mindspore.train.model import Model, ParallelMode
from mindspore.communication.management import init
from mindspore.profiler.profiling import Profiler
from mindspore.train.serialization import load_checkpoint

from src.vit import get_network
from src.dataset import get_dataset
from src.optimizer import get_optimizer
from src.eval_engine import get_eval_engine
from src.logging import get_logger
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper

try:
    os.environ['MINDSPORE_HCCL_CONFIG_PATH'] = os.getenv('RANK_TABLE_FILE')

    device_id = int(os.getenv('DEVICE_ID'))   # 0 ~ 7
    local_rank = int(os.getenv('RANK_ID'))    # local_rank
    device_num = int(os.getenv('RANK_SIZE'))  # world_size
    print("distribute")
except TypeError:
    device_id = 0   # 0 ~ 7
    local_rank = 0    # local_rank
    device_num = 1  # world_size
    print("standalone")

def add_static_args(args):
    """add_static_args"""
    args.train_image_size = args.eval_image_size
    args.weight_decay = 0.05
    args.no_weight_decay_filter = ""
    args.gc_flag = 0
    args.beta1 = 0.9
    args.beta2 = 0.999
    args.loss_scale = 1024

    args.dataset_name = 'imagenet'
    args.save_checkpoint_path = './outputs'
    args.eval_engine = 'imagenet'
    args.auto_tune = 0
    args.seed = 1

    args.device_id = device_id
    args.local_rank = local_rank
    args.device_num = device_num

    return args

@moxing_wrapper()
def eval_net():
    """eval_net"""
    args = add_static_args(config)
    np.random.seed(args.seed)
    args.logger = get_logger(args.save_checkpoint_path, rank=local_rank)

    context.set_context(device_id=device_id,
                        mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        save_graphs=False)

    if args.auto_tune:
        context.set_context(auto_tune_mode='GA')
    elif args.device_num == 1:
        pass
    else:
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)

    if args.open_profiler:
        profiler = Profiler(output_path="data_{}".format(local_rank))

    # init the distribute env
    if not args.auto_tune and args.device_num > 1:
        init()

    # network
    net = get_network(backbone_name=args.backbone, args=args)

    if os.path.isfile(args.pretrained):
        load_checkpoint(args.pretrained, net, strict_load=False)

    # evaluation dataset
    eval_dataset = get_dataset(dataset_name=args.dataset_name,
                               do_train=False,
                               dataset_path=args.eval_path,
                               args=args)

    opt, _ = get_optimizer(optimizer_name='adamw',
                           network=net,
                           lrs=1.0,
                           args=args)

    # evaluation engine
    if args.auto_tune or args.open_profiler or eval_dataset is None:
        args.eval_engine = ''
    eval_engine = get_eval_engine(args.eval_engine, net, eval_dataset, args)

    # model
    model = Model(net, loss_fn=None, optimizer=opt,
                  metrics=eval_engine.metric, eval_network=eval_engine.eval_network,
                  loss_scale_manager=None, amp_level="O3")
    eval_engine.set_model(model)
    args.logger.save_args(args)
    eval_engine.compile(sink_size=625) #step_size

    eval_engine.eval()
    output = eval_engine.get_result()

    print_str = 'accuracy={:.6f}'.format(float(output))
    print(print_str)

    if args.open_profiler:
        profiler.analyse()

if __name__ == '__main__':
    eval_net()
