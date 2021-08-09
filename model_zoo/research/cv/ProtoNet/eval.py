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
"""
ProtoNet evaluation script.
"""
import numpy as np
from mindspore import dataset as ds
from mindspore import load_checkpoint
import mindspore.context as context
from src.protonet import ProtoNet
from src.parser_util import get_parser
from src.PrototypicalLoss import PrototypicalLoss
from model_init import init_dataloader
from train import WithLossCell


def test(test_dataloader, net):
    """
    test function
    """
    inp = ds.GeneratorDataset(test_dataloader, column_names=['data', 'label', 'classes'])
    avg_acc = list()
    avg_loss = list()
    for _ in range(10):
        i = 0
        for batch in inp.create_dict_iterator():
            i = i + 1
            print(i)
            x = batch['data']
            y = batch['label']
            classes = batch['classes']
            acc, loss = net(x, y, classes)
            avg_acc.append(acc.asnumpy())
            avg_loss.append(loss.asnumpy())
    print('eval end')
    avg_acc = np.mean(avg_acc)
    avg_loss = np.mean(avg_loss)
    print('Test Acc: {}  Loss: {}'.format(avg_acc, avg_loss))

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE)
    options = get_parser().parse_args()
    if options.run_offline:
        datapath = options.dataset_root
        ckptpath = options.experiment_root
    else:
        import mox
        mox.file.copy_parallel(src_url=options.data_url, dst_url='cache/data')
        mox.file.copy_parallel(src_url=options.ckpt_url, dst_url='cache/ckpt')
        datapath = 'cache/data'
        ckptpath = 'cache/ckpt'
    Net = ProtoNet()
    loss_fn = PrototypicalLoss(options.num_support_val, options.num_query_val,
                               options.classes_per_it_val, is_train=False)
    Net = WithLossCell(Net, loss_fn)
    val_dataloader = init_dataloader(options, 'val', datapath)
    load_checkpoint(ckptpath, net=Net)
    test(val_dataloader, Net)
