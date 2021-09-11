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
"""eval resnet."""
import os
import ast
import argparse
import numpy as np
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.resnet import resnet50
from src.dataset import create_dataset0 as create_dataset
from src.utility import GetDatasetGenerator_eval, recall_topk_parallel

parser = argparse.ArgumentParser(description='Image classification')
# modelarts parameter
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--ckpt_url', type=str, default=None, help='ckpt path')
parser.add_argument('--checkpoint_name', type=str, default='resnet-120_625.ckpt', help='Checkpoint file')
# Ascend parameter
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--ckpt_path', type=str, default=None, help='Checkpoint file path')
parser.add_argument('--device_id', type=int, default=0, help='Device id')
parser.add_argument('--run_modelarts', type=ast.literal_eval, default=False, help='Run distribute')
args_opt = parser.parse_args()
set_seed(1)

if __name__ == '__main__':

    if args_opt.run_modelarts:
        import moxing as mox
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        context.set_context(device_id=device_id)
        local_data_url = '/cache/data/'
        local_ckpt_url = '/cache/ckpt/'
        mox.file.copy_parallel(args_opt.data_url, local_data_url)
        mox.file.copy_parallel(args_opt.ckpt_url, local_ckpt_url)
        DATA_DIR = local_data_url
    else:
        device_id = args_opt.device_id
        device_num = 1
        context.set_context(device_id=args_opt.device_id)
        DATA_DIR = args_opt.dataset_path

    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', save_graphs=False)

    #dataset
    VAL_LIST = DATA_DIR + "/test_half.txt"
    dataset_generator_val = GetDatasetGenerator_eval(DATA_DIR, VAL_LIST)

    eval_dataset = create_dataset(dataset_generator_val, do_train=False, batch_size=30,
                                  device_num=device_num, rank_id=device_id)
    step_size = eval_dataset.get_dataset_size()

    # define net
    net = resnet50(class_num=5184)

    # load checkpoint
    if args_opt.run_modelarts:
        checkpoint_path = os.path.join(local_ckpt_url, args_opt.checkpoint_name)
    else:
        checkpoint_path = args_opt.ckpt_path
    param_dict = load_checkpoint(checkpoint_path)
    load_param_into_net(net.backbone, param_dict)
    net.set_train(False)

    # define  model
    model_eval = Model(net.backbone)
    f, l = [], []
    for data in eval_dataset.create_dict_iterator():
        out = model_eval.predict(data['image'])
        f.append(out.asnumpy())
        l.append(data['label'].asnumpy())
    f = np.vstack(f)
    l = np.hstack(l)
    recall = recall_topk_parallel(f, l, k=1)
    print("eval_recall:", recall, "ckpt=", checkpoint_path)
