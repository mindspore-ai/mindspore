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
# ===========================================================================
"""train_criteo."""
import argparse
import os

from src.callback import AUCCallBack
from src.callback import TimeMonitor, LossCallback
from src.config import ModelConfig
from src.dataset import get_mindrecord_dataset
from src.fat_deepffm import ModelBuilder
from src.metrics import AUCMetric
from mindspore import context, Model
from mindspore.common import set_seed
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint

parser = argparse.ArgumentParser(description='CTR Prediction')
parser.add_argument('--dataset_path', type=str, default="./data/mindrecord", help='Dataset path')
parser.add_argument('--ckpt_path', type=str, default="Fat-DeepFFM", help='Checkpoint path')
parser.add_argument('--eval_file_name', type=str, default="./auc.log",
                    help='Auc log file path. Default: "./auc.log"')
parser.add_argument('--loss_file_name', type=str, default="./loss.log",
                    help='Loss log file path. Default: "./loss.log"')
parser.add_argument('--device_target', type=str, default="Ascend", choices=("Ascend", "GPU", "CPU"),
                    help="device target, support Ascend, GPU and CPU.")
parser.add_argument('--do_eval', type=bool, default=False,
                    help="Whether side training changes verification.")
args = parser.parse_args()
rank_size = int(os.environ.get("RANK_SIZE", 1))
print("rank_size", rank_size)

set_seed(1)

if __name__ == '__main__':
    model_config = ModelConfig()
    if rank_size > 1:
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target,
                            device_id=device_id)
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True,
                                          all_reduce_fusion_config=[9, 11])
        init()
        rank_id = get_rank()
    else:
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target,
                            device_id=device_id)
        rank_size = None
        rank_id = None
    print("load dataset...")
    ds_train = get_mindrecord_dataset(args.dataset_path, train_mode=True, epochs=1, batch_size=model_config.batch_size,
                                      rank_size=rank_size, rank_id=rank_id, line_per_sample=1000)
    train_net, test_net = ModelBuilder(model_config).get_train_eval_net()
    auc_metric = AUCMetric()
    model = Model(train_net, eval_network=test_net, metrics={"AUC": auc_metric})
    time_callback = TimeMonitor(data_size=ds_train.get_dataset_size())
    loss_callback = LossCallback(args.loss_file_name)
    cb = [loss_callback, time_callback]
    if rank_size == 1 or device_id == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size() * model_config.epoch_size,
                                     keep_checkpoint_max=model_config.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix=args.ckpt_path, config=config_ck)
        cb += [ckpoint_cb]
    if args.do_eval and device_id == 0:
        ds_test = get_mindrecord_dataset(args.dataset_path, train_mode=False)
        eval_callback = AUCCallBack(model, ds_test, eval_file_path=args.eval_file_name)
        cb.append(eval_callback)
    print("Training started...")
    model.train(model_config.epoch_size, train_dataset=ds_train, callbacks=cb, dataset_sink_mode=True)
