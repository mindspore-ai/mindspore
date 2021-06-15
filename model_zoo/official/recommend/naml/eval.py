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
"""Evaluation NAML."""
import os
import time
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint
from src.naml import NAML, NAMLWithLossCell
from src.dataset import MINDPreprocess
from src.utils import NAMLMetric, get_metric

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


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_eval():
    """run eval."""
    config.phase = "eval"
    config.neg_sample = config.eval_neg_sample
    config.rank = get_rank_id()
    config.device_id = get_device_id()
    config.device_num = get_device_num()
    context.set_context(mode=context.GRAPH_MODE, device_target=config.platform, device_id=config.device_id,
                        save_graphs=config.save_graphs, save_graphs_path="naml_ir")

    config.embedding_file = os.path.join(config.dataset_path, config.embedding_file)
    config.word_dict_path = os.path.join(config.dataset_path, config.word_dict_path)
    config.category_dict_path = os.path.join(config.dataset_path, config.category_dict_path)
    config.subcategory_dict_path = os.path.join(config.dataset_path, config.subcategory_dict_path)
    config.uid2index_path = os.path.join(config.dataset_path, config.uid2index_path)
    config.train_dataset_path = os.path.join(config.dataset_path, config.train_dataset_path)
    config.eval_dataset_path = os.path.join(config.dataset_path, config.eval_dataset_path)

    set_seed(config.seed)
    net = NAML(config)
    net.set_train(False)
    net_with_loss = NAMLWithLossCell(net)
    load_checkpoint(config.checkpoint_path, net_with_loss)
    news_encoder = net.news_encoder
    user_encoder = net.user_encoder
    metric = NAMLMetric()
    mindpreprocess = MINDPreprocess(vars(config), dataset_path=config.eval_dataset_path)
    get_metric(config, mindpreprocess, news_encoder, user_encoder, metric)

if __name__ == '__main__':
    run_eval()
