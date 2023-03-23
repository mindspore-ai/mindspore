# Copyright 2022 Huawei Technologies Co., Ltd
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
"""backup and restore related classes and functions."""
from __future__ import absolute_import

import os
import stat

from mindspore import log as logger
from mindspore.train.serialization import load_checkpoint, save_checkpoint
from mindspore.train.callback._callback import Callback
from mindspore.train._utils import _make_directory
from mindspore._checkparam import Validator


class BackupAndRestore(Callback):
    """
    Callback to back up and restore the parameters during training.

    Note:
        This function can only use in training.

    Args:
        backup_dir (str): Path to store and load the checkpoint file.
        save_freq(Union['epoch', int]): When set to 'epoch' the callback saves the checkpoint at the end of
                                        each epoch. When set to an integer, the callback saves the checkpoint
                                        every `save_freq` epoch. Default: 'epoch'.
        delete_checkpoint(bool): If `delete_checkpoint=True`, the checkpoint will be deleted after
                                        training is finished. Default: True.

    Raises:
        ValueError: If backup_dir is not str.
        ValueError: If save_freq is not 'epoch' or int.
        ValueError: If delete_checkpoint is not bool.

    Examples:
        .. note::
            Before running the following example, you need to customize the network LeNet5 and
            dataset preparation function create_dataset. Refer to
            `Building a Network <https://www.mindspore.cn/tutorials/en/master/beginner/model.html>`_
            and `Dataset <https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html>`_ .

        >>> from mindspore import nn
        >>> from mindspore.train import Model, BackupAndRestore
        >>>
        >>> net = LeNet5()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        >>> optim = nn.Momentum(net.trainable_params(), 0.01, 0.9)
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
        >>> data_path = './MNIST_Data'
        >>> dataset = create_dataset(data_path)
        >>> backup_ckpt = BackupAndRestore("backup")
        >>> model.train(10, dataset, callbacks=backup_ckpt)
    """
    def __init__(self, backup_dir, save_freq="epoch", delete_checkpoint=True):
        super(BackupAndRestore, self).__init__()
        ckpt_dir = _make_directory(backup_dir)
        self.backup_file = os.path.join(ckpt_dir, 'backup.ckpt')
        if save_freq != "epoch":
            self.save_freq = Validator.check_positive_int(save_freq)
        else:
            self.save_freq = 1
        self.delete_checkpoint = Validator.check_bool(delete_checkpoint)

    def on_train_begin(self, run_context):
        """
        Load the backup checkpoint file at the beginning of epoch.

        Args:
            run_context (RunContext): Context of the process running. For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        if os.path.exists(self.backup_file):
            cb_params = run_context.original_args()
            train_net = cb_params.train_network
            logger.info("Restore checkpoint file is {}, load checkpoint into train net.".format(self.backup_file))
            load_checkpoint(self.backup_file, net=train_net)

    def on_train_epoch_end(self, run_context):
        """
        Backup checkpoint file at the end of train epoch.

        Args:
           run_context (RunContext): Context of the process running. For more details,
                   please refer to :class:`mindspore.train.RunContext`.
        """
        cb_params = run_context.original_args()
        cur_epoch_num = cb_params.cur_epoch_num
        if cur_epoch_num % self.save_freq == 0:
            train_net = cb_params.train_network
            logger.info("Train task end, backup checkpoint file: {}.".format(self.backup_file))
            save_checkpoint(train_net, self.backup_file)

    def on_train_end(self, run_context):
        """
        Deleted checkpoint file at the end of train.

        Args:
            run_context (RunContext): Context of the process running. For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        run_context.original_args()
        cb_params = run_context.original_args()
        cur_epoch_num = cb_params.cur_epoch_num
        if self.delete_checkpoint:
            logger.info("Delete restore checkpoint file {} at {} epoch.".format(self.backup_file, cur_epoch_num))
            os.chmod(self.backup_file, stat.S_IWRITE)
            os.remove(self.backup_file)
