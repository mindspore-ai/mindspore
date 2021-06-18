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
import os
import math
from mindspore import nn, load_checkpoint, context
import mindspore.common.dtype as mstype
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.context import ParallelMode
from mindspore.communication.management import init
from src.naml import NAML, NAMLWithLossCell
from src.dataset import create_dataset, MINDPreprocess
from src.utils import process_data
from src.callback import Monitor

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_rank_id, get_device_id, get_device_num


def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))

    config.save_checkpoint_path = os.path.join(config.output_path, config.save_checkpoint_path)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    """run train."""
    config.phase = "train"
    config.rank = get_rank_id()
    config.device_id = get_device_id()
    config.device_num = get_device_num()
    if config.device_num > 1:
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform, save_graphs=config.save_graphs)
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=config.device_num)
        init()
        config.save_checkpoint_path = os.path.join(config.save_checkpoint_path, "ckpt_" + str(config.rank))
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=config.platform, device_id=config.device_id,
                            save_graphs=config.save_graphs, save_graphs_path="naml_ir")

    config.epochs = config.default_epochs * math.ceil(config.device_num ** 0.5) if config.epochs <= 0 else config.epochs

    config.embedding_file = os.path.join(config.dataset_path, config.embedding_file)
    config.word_dict_path = os.path.join(config.dataset_path, config.word_dict_path)
    config.category_dict_path = os.path.join(config.dataset_path, config.category_dict_path)
    config.subcategory_dict_path = os.path.join(config.dataset_path, config.subcategory_dict_path)
    config.uid2index_path = os.path.join(config.dataset_path, config.uid2index_path)
    config.train_dataset_path = os.path.join(config.dataset_path, config.train_dataset_path)
    config.eval_dataset_path = os.path.join(config.dataset_path, config.eval_dataset_path)

    set_seed(config.seed)
    word_embedding = process_data(config)
    net = NAML(config, word_embedding)
    net_with_loss = NAMLWithLossCell(net)
    if config.checkpoint_path:
        load_checkpoint(config.pretrain_checkpoint, net_with_loss)
    mindpreprocess_train = MINDPreprocess(vars(config), dataset_path=config.train_dataset_path)
    dataset = create_dataset(mindpreprocess_train, batch_size=config.batch_size, rank=config.rank,
                             group_size=config.device_num)
    config.dataset_size = dataset.get_dataset_size()
    config.print_times = min(config.dataset_size, config.print_times)
    if config.weight_decay:
        weight_params = list(filter(lambda x: 'weight' in x.name, net.trainable_params()))
        other_params = list(filter(lambda x: 'weight' not in x.name, net.trainable_params()))
        group_params = [{'params': weight_params, 'weight_decay': 1e-3},
                        {'params': other_params, 'weight_decay': 0.0},
                        {'order_params': net.trainable_params()}]
        opt = nn.AdamWeightDecay(group_params, config.lr, beta1=config.beta1, beta2=config.beta2, eps=config.epsilon)
    else:
        opt = nn.Adam(net.trainable_params(), config.lr, beta1=config.beta1, beta2=config.beta2, eps=config.epsilon)
    if config.mixed:
        loss_scale_manager = DynamicLossScaleManager(init_loss_scale=128.0, scale_factor=2, scale_window=10000)
        net_with_loss.to_float(mstype.float16)
        for _, cell in net_with_loss.cells_and_names():
            if isinstance(cell, (nn.Embedding, nn.Softmax, nn.SoftmaxCrossEntropyWithLogits)):
                cell.to_float(mstype.float32)
        model = Model(net_with_loss, optimizer=opt, loss_scale_manager=loss_scale_manager)
    else:
        model = Model(net_with_loss, optimizer=opt)
    cb = [Monitor(config)]
    epochs = config.epochs
    if config.sink_mode:
        epochs = int(config.epochs * config.dataset_size / config.print_times)
    start_time = time.time()
    print("======================= Start Train ==========================", flush=True)
    model.train(epochs, dataset, callbacks=cb, dataset_sink_mode=config.sink_mode, sink_size=config.print_times)
    end_time = time.time()
    print("==============================================================")
    print("processor_name: {}".format(config.platform))
    print("test_name: NAML")
    print(f"model_name: NAML MIND{config.dataset}")
    print("batch_size: {}".format(config.batch_size))
    print("latency: {} s".format(end_time - start_time))

if __name__ == '__main__':
    run_train()
