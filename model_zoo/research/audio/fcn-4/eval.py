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
'''
##############evaluate trained models#################
python eval.py
'''

import argparse
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.musictagger import MusicTaggerCNN
from src.config import music_cfg as cfg
from src.dataset import create_dataset


def calculate_auc(labels_list, preds_list):
    """
    The AUC calculation function
    Input:
            labels_list: list of true label
            preds_list:  list of predicted label
    Outputs
            Float, means of AUC
    """
    auc = []
    n_bins = labels_list.shape[0] // 2
    if labels_list.ndim == 1:
        labels_list = labels_list.reshape(-1, 1)
        preds_list = preds_list.reshape(-1, 1)
    for i in range(labels_list.shape[1]):
        labels = labels_list[:, i]
        preds = preds_list[:, i]
        postive_len = labels.sum()
        negative_len = labels.shape[0] - postive_len
        total_case = postive_len * negative_len
        positive_histogram = np.zeros((n_bins))
        negative_histogram = np.zeros((n_bins))
        bin_width = 1.0 / n_bins

        for j, _ in enumerate(labels):
            nth_bin = int(preds[j] // bin_width)
            if labels[j]:
                positive_histogram[nth_bin] = positive_histogram[nth_bin] + 1
            else:
                negative_histogram[nth_bin] = negative_histogram[nth_bin] + 1

        accumulated_negative = 0
        satisfied_pair = 0
        for k in range(n_bins):
            satisfied_pair += (
                positive_histogram[k] * accumulated_negative +
                positive_histogram[k] * negative_histogram[k] * 0.5)
            accumulated_negative += negative_histogram[k]
        auc.append(satisfied_pair / total_case)

    return np.mean(auc)


def val(net, data_dir, filename, num_consumer=4, batch=32):
    """
    Validation function, estimate the performance of trained model

    Input:
            net:        the trained neural network
            data_dir:       path to the validation dataset
            filename:       name of the validation dataset
            num_consumer:   split number of validation dataset
            batch:          validation batch size
    Outputs
            Float, AUC
    """
    data_train = create_dataset(data_dir, filename, 32, ['feature', 'label'],
                                num_consumer)
    data_train = data_train.create_tuple_iterator()
    res_pred = []
    res_true = []
    for data, label in data_train:
        x = net(Tensor(data, dtype=mstype.float32))
        res_pred.append(x.asnumpy())
        res_true.append(label.asnumpy())
    res_pred = np.concatenate(res_pred, axis=0)
    res_true = np.concatenate(res_true, axis=0)
    auc = calculate_auc(res_true, res_pred)
    return auc


def validation(net, model_path, data_dir, filename, num_consumer, batch):
    param_dict = load_checkpoint(model_path)
    load_param_into_net(net, param_dict)

    auc = val(net, data_dir, filename, num_consumer, batch)
    return auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--device_id',
                        type=int,
                        help='device ID',
                        default=None)
    args = parser.parse_args()

    if args.device_id is not None:
        context.set_context(device_target=cfg.device_target,
                            mode=context.GRAPH_MODE,
                            device_id=args.device_id)
    else:
        context.set_context(device_target=cfg.device_target,
                            mode=context.GRAPH_MODE,
                            device_id=cfg.device_id)

    network = MusicTaggerCNN(in_classes=[1, 128, 384, 768, 2048],
                             kernel_size=[3, 3, 3, 3, 3],
                             padding=[0] * 5,
                             maxpool=[(2, 4), (4, 5), (3, 8), (4, 8)],
                             has_bias=True)
    network.set_train(False)
    auc_val = validation(network, cfg.checkpoint_path + "/" + cfg.model_name, cfg.data_dir,
                         cfg.val_filename, cfg.num_consumer, cfg.batch_size)

    print("=" * 10 + "Validation Peformance" + "=" * 10)
    print("AUC: {:.5f}".format(auc_val))
