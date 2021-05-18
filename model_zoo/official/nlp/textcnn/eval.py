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
##############test textcnn example on movie review#################
python eval.py
"""
import mindspore.nn as nn
from mindspore.nn.metrics import Accuracy
from mindspore import context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id
from model_utils.config import config
from src.textcnn import TextCNN
from src.dataset import MovieReview, SST2, Subjectivity

@moxing_wrapper()
def eval_net():
    '''eval net'''
    if config.dataset == 'MR':
        instance = MovieReview(root_dir=config.data_path, maxlen=config.word_len, split=0.9)
    elif config.dataset == 'SUBJ':
        instance = Subjectivity(root_dir=config.data_path, maxlen=config.word_len, split=0.9)
    elif config.dataset == 'SST2':
        instance = SST2(root_dir=config.data_path, maxlen=config.word_len, split=0.9)
    device_target = config.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if device_target == "Ascend":
        context.set_context(device_id=get_device_id())
    dataset = instance.create_test_dataset(batch_size=config.batch_size)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    net = TextCNN(vocab_len=instance.get_dict_len(), word_len=config.word_len,
                  num_classes=config.num_classes, vec_length=config.vec_length)
    opt = nn.Adam(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate=0.001,
                  weight_decay=float(config.weight_decay))

    param_dict = load_checkpoint(config.checkpoint_file_path)
    print("load checkpoint from [{}].".format(config.checkpoint_file_path))

    load_param_into_net(net, param_dict)
    net.set_train(False)
    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc': Accuracy()})

    acc = model.eval(dataset)
    print("accuracy: ", acc)

if __name__ == '__main__':
    eval_net()
