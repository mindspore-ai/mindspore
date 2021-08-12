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
"""import"""
import os
from src.models.roberta_one_sent_classification_en import RobertaOneSentClassificationEn
from src.config import bertcfg
from src.utils import args
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context
from mindspore.dataset import MindDataset
import numpy as np


def model_from_params(params_dict, is_training=True):
    """
    :param params_dict:
    :return:
    """
    model = RobertaOneSentClassificationEn(params_dict, is_training)
    return model


if __name__ == "__main__":
    args = args.build_common_arguments()
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target="Ascend",
        device_id=0)
    task_name = args.job
    MINDRECORD_FILE_PATH = os.path.join(args.data_url, task_name, task_name + "_dev.mindrecord")
    ds_eval = MindDataset(MINDRECORD_FILE_PATH, num_parallel_workers=8).batch(args.batch_size, True,
                                                                              8).create_dict_iterator()

    model_eval = model_from_params(bertcfg, False).bert
    param_dict = load_checkpoint(args.ckpt)
    load_param_into_net(model_eval, param_dict)
    total = 0
    total_acc = 0
    for i in ds_eval:
        logits = model_eval(
            i['src_ids'],
            i['sent_ids'],
            i['mask_ids']).asnumpy()
        preds = np.argmax(logits, axis=1).reshape(-1)
        labels = i['label'].asnumpy().reshape(-1)
        acc = (preds == labels).sum()
        total += len(preds)
        total_acc += acc
    print(total_acc / total)
