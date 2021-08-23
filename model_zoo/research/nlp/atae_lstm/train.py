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
"""Train"""
import os
import argparse
import numpy as np

from mindspore import nn, Tensor
from mindspore import context
from mindspore import set_seed
from mindspore import Model
from mindspore.common import dtype as mstype
from mindspore import save_checkpoint
from mindspore.train.callback import Callback
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.callback import TimeMonitor, LossMonitor, SummaryCollector

from src.atae_for_train import NetWithLoss
from src.model import AttentionLstm
from src.load_dataset import load_dataset
from src.config import Config


class SaveCallback(Callback):
    """
    define savecallback, save best model while training.
    """
    def __init__(self, eval_model, dataset, save_file_path):
        super(SaveCallback, self).__init__()
        self.model = Model(eval_model)
        self.eval_dataset = dataset
        self.save_path = save_file_path
        self.acc = 0.8

    def step_end(self, run_context):
        """
        eval and save ckpt while training
        """
        cb_params = run_context.original_args()

        correct = 0
        count = 0
        for batch in self.eval_dataset.create_dict_iterator():
            content = batch['content']
            sen_len = batch['sen_len']
            aspect = batch['aspect']
            solution = batch['solution']

            pred = self.model.predict(content, sen_len, aspect)

            polarity_pred = np.argmax(pred.asnumpy())
            polarity_label = np.argmax(solution.asnumpy())
            if polarity_pred == polarity_label:
                correct += 1
            count += 1
        res = correct / count

        if res > self.acc:
            self.acc = res
            file_name = self.save_path + '_max' + ".ckpt"
            save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=file_name)
            print("Save the maximum accuracy checkpoint,the accuracy is", self.acc)


def get_config(config_json):
    """get atae_lstm configuration"""
    cfg = Config.from_json_file(config_json)
    return cfg


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ATAE_LSTM train entry point.')
    parser.add_argument("--config", type=str, required=True, help="configuration address.")
    parser.add_argument("--data_url", type=str, required=True, help="input dataset.")
    parser.add_argument("--train_url", type=str, required=True, help="output dataset.")
    parser.add_argument("--is_modelarts", type=bool, default=False, help="is modelarts platform")

    args, _ = parser.parse_known_args()

    config = get_config(args.config)

    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        save_graphs=False)

    set_seed(config.rseed)

    data_menu = args.data_url

    if args.is_modelarts:
        import moxing as mox
        mox.file.copy_parallel(src_url=args.data_url, dst_url='/cache/dataset_menu')
        data_menu = '/cache/dataset_menu/'

    train_dataset = data_menu + 'train.mindrecord'
    eval_dataset = data_menu + 'test.mindrecord'
    word_path = data_menu + 'weight.npz'

    dataset_train = load_dataset(input_files=train_dataset,
                                 batch_size=config.batch_size)
    dataset_val = load_dataset(input_files=eval_dataset,
                               batch_size=config.batch_size)

    r = np.load(word_path)
    word_vector = r['weight']
    weight = Tensor(word_vector, mstype.float16)

    net = AttentionLstm(config, weight, is_train=True)
    model_with_loss = NetWithLoss(net, batch_size=1)

    epoch_size = 25
    step_per_epoch = dataset_train.get_dataset_size()
    optimizer = nn.Momentum(params=net.trainable_params(),
                            learning_rate=config.lr,
                            momentum=config.momentum,
                            weight_decay=config.weight_decay)

    train_net = nn.TrainOneStepCell(model_with_loss, optimizer)

    model = Model(train_net)

    time_cb = TimeMonitor(data_size=step_per_epoch)
    loss_cb = LossMonitor()
    summary_collector = SummaryCollector(summary_dir='./train/summary_dir')
    cb = [time_cb, loss_cb, summary_collector]
    config_ck = CheckpointConfig(save_checkpoint_steps=step_per_epoch,
                                 keep_checkpoint_max=epoch_size)
    ckpoint_cb = ModelCheckpoint(prefix="atae-lstm", directory='./train/', config=config_ck)
    cb.append(ckpoint_cb)

    ckpt_path = './train/atae-lstm'
    if args.is_modelarts:
        os.makedirs('/cache/train_output/')
        ckpt_path = '/cache/train_output/atae-lstm'
    save_cb = SaveCallback(net, dataset_val, ckpt_path)
    cb.append(save_cb)

    print("start train")
    model.train(epoch_size, dataset_train, callbacks=cb)
    print("train success!")

    if args.is_modelarts:
        mox.file.copy_parallel(src_url='/cache/train_output/', dst_url=args.train_url)
