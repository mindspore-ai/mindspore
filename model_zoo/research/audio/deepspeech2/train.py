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

"""train_criteo."""
import os
import json
import argparse

from mindspore import context, Tensor, ParameterTuple
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn.optim import Adam
from mindspore.nn import TrainOneStepCell
from mindspore.train import Model

from src.deepspeech2 import DeepSpeechModel, NetWithLossClass
from src.lr_generator import get_lr
from src.config import train_config
from src.dataset import create_dataset

parser = argparse.ArgumentParser(description='DeepSpeech2 training')
parser.add_argument('--pre_trained_model_path', type=str, default='', help='Pretrained checkpoint path')
parser.add_argument('--is_distributed', action="store_true", default=False, help='Distributed training')
parser.add_argument('--bidirectional', action="store_false", default=True, help='Use bidirectional RNN')
parser.add_argument('--device_target', type=str, default="GPU", choices=("GPU", "CPU"),
                    help='Device target, support GPU and CPU, Default: GPU')
args = parser.parse_args()

if __name__ == '__main__':
    rank_id = 0
    group_size = 1
    config = train_config
    data_sink = (args.device_target == "GPU")
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=False)
    if args.is_distributed:
        init('nccl')
        rank_id = get_rank()
        group_size = get_group_size()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)

    with open(config.DataConfig.labels_path) as label_file:
        labels = json.load(label_file)

    ds_train = create_dataset(audio_conf=config.DataConfig.SpectConfig,
                              manifest_filepath=config.DataConfig.train_manifest,
                              labels=labels, normalize=True, train_mode=True,
                              batch_size=config.DataConfig.batch_size, rank=rank_id, group_size=group_size)
    steps_size = ds_train.get_dataset_size()

    lr = get_lr(lr_init=config.OptimConfig.learning_rate, total_epochs=config.TrainingConfig.epochs,
                steps_per_epoch=steps_size)
    lr = Tensor(lr)

    deepspeech_net = DeepSpeechModel(batch_size=config.DataConfig.batch_size,
                                     rnn_hidden_size=config.ModelConfig.hidden_size,
                                     nb_layers=config.ModelConfig.hidden_layers,
                                     labels=labels,
                                     rnn_type=config.ModelConfig.rnn_type,
                                     audio_conf=config.DataConfig.SpectConfig,
                                     bidirectional=True,
                                     device_target=args.device_target)

    loss_net = NetWithLossClass(deepspeech_net)
    weights = ParameterTuple(deepspeech_net.trainable_params())

    optimizer = Adam(weights, learning_rate=config.OptimConfig.learning_rate, eps=config.OptimConfig.epsilon,
                     loss_scale=config.OptimConfig.loss_scale)
    train_net = TrainOneStepCell(loss_net, optimizer)
    train_net.set_train(True)
    if args.pre_trained_model_path != '':
        param_dict = load_checkpoint(args.pre_trained_model_path)
        load_param_into_net(train_net, param_dict)
        print('Successfully loading the pre-trained model')

    model = Model(train_net)
    callback_list = [TimeMonitor(steps_size), LossMonitor()]

    if args.is_distributed:
        config.CheckpointConfig.ckpt_file_name_prefix = config.CheckpointConfig.ckpt_file_name_prefix + str(get_rank())
        config.CheckpointConfig.ckpt_path = os.path.join(config.CheckpointConfig.ckpt_path,
                                                         'ckpt_' + str(get_rank()) + '/')
    config_ck = CheckpointConfig(save_checkpoint_steps=1,
                                 keep_checkpoint_max=config.CheckpointConfig.keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint(prefix=config.CheckpointConfig.ckpt_file_name_prefix,
                              directory=config.CheckpointConfig.ckpt_path, config=config_ck)
    callback_list.append(ckpt_cb)
    model.train(config.TrainingConfig.epochs, ds_train, callbacks=callback_list, dataset_sink_mode=data_sink)
