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
GCN training script.
"""
import os
import time
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.common.dtype as mstype
from mindspore import Model, context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor, LossMonitor
from mindspore import Tensor
import numpy as np

from src.gcn import GCN
from src.metrics import Loss, GCNAccuracy, apply_eval
from src.config import ConfigGCN
from src.dataset import get_adj_features_labels, get_mask
from src.eval_callback import EvalCallBack

from model_utils.config import config as default_args
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num


def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, default_args.modelarts_dataset_unzip_name)):
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

    if default_args.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(default_args.data_path, default_args.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(default_args.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if default_args.device_target == "Ascend":
            device_id = get_device_id()
            device_num = get_device_num()
        else:
            raise ValueError("Not support device_target.")

        if device_id % min(device_num, 8) == 0 and not os.path.exists(sync_lock):
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

        print("Device: {}, Finish sync unzip data from {} to {}.".format(device_id, zip_file_1, save_dir_1))

    default_args.save_ckptpath = os.path.join(default_args.output_path, default_args.save_ckptpath)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    """Train model."""
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=default_args.device_target, save_graphs=False)
    config = ConfigGCN()
    if not os.path.exists(config.ckpt_dir):
        os.mkdir(config.ckpt_dir)
    adj, feature, label_onehot, _ = get_adj_features_labels(default_args.data_dir)
    feature_d = np.expand_dims(feature, axis=0)
    label_onehot_d = np.expand_dims(label_onehot, axis=0)
    data = {"feature": feature_d, "label": label_onehot_d}
    dataset = ds.NumpySlicesDataset(data=data)
    nodes_num = label_onehot.shape[0]
    eval_mask = get_mask(nodes_num, default_args.train_nodes_num,
                         default_args.train_nodes_num + default_args.eval_nodes_num)
    class_num = label_onehot.shape[1]
    input_dim = feature.shape[1]
    adj = Tensor(adj, dtype=mstype.float32)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=config.save_ckpt_steps,
                                   keep_checkpoint_max=config.keep_ckpt_max)
    ckpoint_cb = ModelCheckpoint(prefix='ckpt_gcn',
                                 directory=config.ckpt_dir,
                                 config=ckpt_config)
    gcn_net = GCN(config, input_dim, class_num, adj)
    cb = [TimeMonitor(), LossMonitor(), ckpoint_cb]
    opt = nn.Adam(gcn_net.trainable_params(), learning_rate=config.learning_rate)
    criterion = Loss(eval_mask, config.weight_decay, gcn_net.trainable_params()[0])
    model = Model(gcn_net, loss_fn=criterion, optimizer=opt, amp_level="O3")
    if default_args.train_with_eval:
        GCN_metric = GCNAccuracy(eval_mask)
        eval_model = Model(gcn_net, loss_fn=criterion, metrics={'GCNAccuracy': GCN_metric})
        eval_param_dict = {"model": eval_model, "dataset": dataset, "metrics_name": "GCNAccuracy"}
        eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=config.eval_interval,
                               eval_start_epoch=default_args.eval_start_epoch, save_best_ckpt=config.save_best_ckpt,
                               ckpt_directory=config.best_ckpt_dir, besk_ckpt_name=config.best_ckpt_name,
                               metrics_name="GCNAccuracy")
        cb.append(eval_cb)
    model.train(config.epochs, dataset, callbacks=cb, dataset_sink_mode=True)

if __name__ == '__main__':
    run_train()
