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
"""Train NAML."""
import time
from mindspore import nn, load_checkpoint
import mindspore.common.dtype as mstype
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from src.naml import NAML, NAMLWithLossCell
from src.option import get_args
from src.dataset import create_dataset, MINDPreprocess
from src.utils import process_data
from src.callback import Monitor

if __name__ == '__main__':
    args = get_args("train")
    set_seed(args.seed)
    word_embedding = process_data(args)
    net = NAML(args, word_embedding)
    net_with_loss = NAMLWithLossCell(net)
    if args.checkpoint_path is not None:
        load_checkpoint(args.pretrain_checkpoint, net_with_loss)
    mindpreprocess_train = MINDPreprocess(vars(args), dataset_path=args.train_dataset_path)
    dataset = create_dataset(mindpreprocess_train, batch_size=args.batch_size, rank=args.rank,
                             group_size=args.device_num)
    args.dataset_size = dataset.get_dataset_size()
    args.print_times = min(args.dataset_size, args.print_times)
    if args.weight_decay:
        weight_params = list(filter(lambda x: 'weight' in x.name, net.trainable_params()))
        other_params = list(filter(lambda x: 'weight' not in x.name, net.trainable_params()))
        group_params = [{'params': weight_params, 'weight_decay': 1e-3},
                        {'params': other_params, 'weight_decay': 0.0},
                        {'order_params': net.trainable_params()}]
        opt = nn.AdamWeightDecay(group_params, args.lr, beta1=args.beta1, beta2=args.beta2, eps=args.epsilon)
    else:
        opt = nn.Adam(net.trainable_params(), args.lr, beta1=args.beta1, beta2=args.beta2, eps=args.epsilon)
    if args.mixed:
        loss_scale_manager = DynamicLossScaleManager(init_loss_scale=128.0, scale_factor=2, scale_window=10000)
        net_with_loss.to_float(mstype.float16)
        for _, cell in net_with_loss.cells_and_names():
            if isinstance(cell, (nn.Embedding, nn.Softmax, nn.SoftmaxCrossEntropyWithLogits)):
                cell.to_float(mstype.float32)
        model = Model(net_with_loss, optimizer=opt, loss_scale_manager=loss_scale_manager)
    else:
        model = Model(net_with_loss, optimizer=opt)
    cb = [Monitor(args)]
    epochs = args.epochs
    if args.sink_mode:
        epochs = int(args.epochs * args.dataset_size / args.print_times)
    start_time = time.time()
    print("======================= Start Train ==========================", flush=True)
    model.train(epochs, dataset, callbacks=cb, dataset_sink_mode=args.sink_mode, sink_size=args.print_times)
    end_time = time.time()
    print("==============================================================")
    print("processor_name: {}".format(args.platform))
    print("test_name: NAML")
    print(f"model_name: NAML MIND{args.dataset}")
    print("batch_size: {}".format(args.batch_size))
    print("latency: {} s".format(end_time - start_time))
