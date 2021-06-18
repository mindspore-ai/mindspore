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
"""Config parameters for skipgram models."""

import os

class ConfigSkipgram:
    """
    Config parameters for the Skipgram.

    Examples:
        ConfigSkipgram()
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    par_dir = os.path.dirname(cur_dir)

    lr = 1e-3                          # initial learning rate
    end_lr = 1e-4                      # end learning rate
    train_epoch = 1                    # training epoch
    data_epoch = 10                    # generate data epoch
    power = 1                          # decay rate of learning rate
    batch_size = 128                   # batch size
    dataset_sink_mode = False
    emb_size = 288                     # embedding size
    min_count = 5                      # keep vocabulary that have appeared at least 'min_count' times
    window_size = 5                    # window size of center word
    neg_sample_num = 5                 # number of negative words in negative sampling
    save_checkpoint_steps = int(5e5)   # step interval between two checkpoints
    keep_checkpoint_max = 15                                    # maximal number of checkpoint files
    temp_dir = os.path.join(par_dir, 'temp/')                   # save files generated during code execution
    ckpt_dir = os.path.join(par_dir, 'temp/ckpts/')             # directory that save checkpoint files
    ms_dir = os.path.join(par_dir, 'temp/ms_dir/')              # directory that saves mindrecord data
    w2v_emb_save_dir = os.path.join(par_dir, 'temp/w2v_emb/')   # directory that saves word2vec embeddings
    train_data_dir = os.path.join(par_dir, 'data/train_data/')  # directory of training corpus
    eval_data_dir = os.path.join(par_dir, 'data/eval_data/')    # directory of evaluating data

w2v_cfg = ConfigSkipgram()
