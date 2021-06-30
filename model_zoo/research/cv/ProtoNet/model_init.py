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
ProtoNet model init script.
"""
import itertools
import mindspore.nn as nn
import numpy as np
from src.dataset import OmniglotDataset
from src.IterDatasetGenerator import IterDatasetGenerator

def init_lr_scheduler(opt):
    '''
    Initialize the learning rate scheduler
    '''
    epochs = opt.epochs
    milestone = list(itertools.takewhile(lambda n: n < epochs, itertools.count(1, opt.lr_scheduler_step)))

    lr0 = opt.learning_rate
    bl = list(np.logspace(0, len(milestone)-1, len(milestone), base=opt.lr_scheduler_gamma))
    lr = [lr0*b for b in bl]
    lr_epoch = nn.piecewise_constant_lr(milestone, lr)
    return lr_epoch

def init_dataset(opt, mode, path):
    '''
    Initialize the dataset
    '''
    dataset = OmniglotDataset(mode=mode, root=path)
    n_classes = len(np.unique(dataset.y))
    if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))
    return dataset

def init_dataloader(opt, mode, path):
    '''
    Initialize the dataloader
    '''
    dataset = init_dataset(opt, mode, path)
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr

    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    dataloader = IterDatasetGenerator(dataset, classes_per_it, num_samples, opt.iterations)
    return dataloader
