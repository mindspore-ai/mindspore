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
BGCF evaluation script.
"""
import os
import time
import datetime

import mindspore.context as context
from mindspore.train.serialization import load_checkpoint
from mindspore.common import set_seed

from src.bgcf import BGCF
from src.utils import BGCFLogger
from src.metrics import BGCFEvaluate
from src.callback import ForwardBGCF, TestBGCF
from src.dataset import TestGraphDataset, load_graph

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num

set_seed(1)

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
def evaluation():
    """evaluation"""
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target,
                        save_graphs=False)
    if config.device_target == "Ascend":
        context.set_context(device_id=get_device_id())

    train_graph, test_graph, sampled_graph_list = load_graph(config.datapath)
    test_graph_dataset = TestGraphDataset(train_graph, sampled_graph_list, num_samples=config.raw_neighs,
                                          num_bgcn_neigh=config.gnew_neighs,
                                          num_neg=config.num_neg)

    if config.log_name:
        now = datetime.datetime.now().strftime("%b_%d_%H_%M_%S")
        name = "bgcf" + '-' + config.log_name + '-' + config.dataset
        log_save_path = './log-files/' + name + '/' + now
        log = BGCFLogger(logname=name, now=now, foldername='log-files', copy=False)
        log.open(log_save_path + '/log.train.txt', mode='a')
        for arg in vars(config):
            log.write(arg + '=' + str(getattr(config, arg)) + '\n')
    else:
        for arg in vars(config):
            print(arg + '=' + str(getattr(config, arg)))

    num_user = train_graph.graph_info()["node_num"][0]
    num_item = train_graph.graph_info()["node_num"][1]

    eval_class = BGCFEvaluate(config, train_graph, test_graph, config.Ks)
    for _epoch in range(config.eval_interval, config.num_epoch+1, config.eval_interval) \
                  if config.device_target == "Ascend" else range(config.num_epoch, config.num_epoch+1):
        bgcfnet_test = BGCF([config.input_dim, num_user, num_item],
                            config.embedded_dimension,
                            config.activation,
                            [0.0, 0.0, 0.0],
                            num_user,
                            num_item,
                            config.input_dim)

        load_checkpoint(config.ckptpath + "/bgcf_epoch{}.ckpt".format(_epoch), net=bgcfnet_test)

        forward_net = ForwardBGCF(bgcfnet_test)
        user_reps, item_reps = TestBGCF(forward_net, num_user, num_item, config.input_dim, test_graph_dataset)

        test_recall_bgcf, test_ndcg_bgcf, \
        test_sedp, test_nov = eval_class.eval_with_rep(user_reps, item_reps, config)

        if config.log_name:
            log.write(
                'epoch:%03d,      recall_@10:%.5f,     recall_@20:%.5f,     ndcg_@10:%.5f,    ndcg_@20:%.5f,   '
                'sedp_@10:%.5f,     sedp_@20:%.5f,    nov_@10:%.5f,    nov_@20:%.5f\n' % (_epoch,
                                                                                          test_recall_bgcf[1],
                                                                                          test_recall_bgcf[2],
                                                                                          test_ndcg_bgcf[1],
                                                                                          test_ndcg_bgcf[2],
                                                                                          test_sedp[0],
                                                                                          test_sedp[1],
                                                                                          test_nov[1],
                                                                                          test_nov[2]))
        else:
            print('epoch:%03d,      recall_@10:%.5f,     recall_@20:%.5f,     ndcg_@10:%.5f,    ndcg_@20:%.5f,   '
                  'sedp_@10:%.5f,     sedp_@20:%.5f,    nov_@10:%.5f,    nov_@20:%.5f\n' % (_epoch,
                                                                                            test_recall_bgcf[1],
                                                                                            test_recall_bgcf[2],
                                                                                            test_ndcg_bgcf[1],
                                                                                            test_ndcg_bgcf[2],
                                                                                            test_sedp[0],
                                                                                            test_sedp[1],
                                                                                            test_nov[1],
                                                                                            test_nov[2]))


if __name__ == "__main__":
    evaluation()
