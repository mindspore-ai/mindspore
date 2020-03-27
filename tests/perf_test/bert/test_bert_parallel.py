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

"""Bert parallel test."""

# pylint: disable=missing-docstring, arguments-differ, W0612
import os
import mindspore.common.dtype as mstype
from mindspore.nn.optim import Lamb
from mindspore.train.model import Model
from mindspore.model_zoo.Bert_NEZHA import BertConfig, BertNetworkWithLoss, BertTrainOneStepWithLossScaleCell
from mindspore import context
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
from mindspore.nn.wrap.loss_scale import LossScaleUpdateCell
from mindspore.train.callback import Callback
from mindspore.train.parallel_utils import ParallelMode
from mindspore import log as logger
import mindspore.communication.management as D
import mindspore.dataset as ds
import mindspore._c_dataengine as deMap


class LossCallBack(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF terminating training.

    Note:
        If per_print_times is 0 do not print loss.

    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, per_print_times=1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        logger.info("epoch: {}, outputs are {}".format(cb_params.cur_epoch_num, str(cb_params.net_outputs)))


def get_config(version='base', batch_size=1):
    if version == 'base':
        return BertConfig(
            batch_size=batch_size,
            seq_length=128,
            vocab_size=21128,
            hidden_size=768,
            num_hidden_layers=24,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            use_relative_positions=True,
            input_mask_from_dataset=True,
            token_type_ids_from_dataset=True,
            dtype=mstype.float32,
            compute_type=mstype.float32)
    elif version == 'large':
        return BertConfig(
            batch_size=batch_size,
            seq_length=128,
            vocab_size=21136,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            use_relative_positions=True,
            input_mask_from_dataset=True,
            token_type_ids_from_dataset=True,
            dtype=mstype.float32,
            compute_type=mstype.float16)
    return BertConfig(batch_size=batch_size)


def gen_fake_distribute_file_json(distribute_file, world_size=1, local_rank=0):
    random_method = 'RANDOM'
    shuffle = 'ON'
    seed = 0

    content = '{\n' \
              '  "deviceNum":%s,\n' \
              '  "deviceId": %s,\n' \
              '  "shardConfig":"%s",\n' \
              '  "shuffle":"%s",\n' \
              '  "seed": %s\n' \
              '}' % (world_size, local_rank, random_method, shuffle, seed)

    with open(distribute_file, 'w') as fw:
        fw.write(content)


def test_gen_fake_minddata():
    local_rank = D.get_rank()
    world_size = D.get_group_size()
    minddata_json = 'minddata_json'
    print('world_size:{}'.format(world_size))
    print('local_rank:{}'.format(local_rank))
    # be careful for 1 machine 8p
    if local_rank % 8 == 0:
        # first remove minddata json and data
        import shutil
        if os.path.exists(minddata_json):
            shutil.rmtree(minddata_json)

        if not os.path.exists(minddata_json):
            os.makedirs(minddata_json)

        # be careful for 1 machine 8p
        for rank in range(world_size):
            distribute_file = os.path.join(minddata_json, 'distribution_{}.json'.format(rank))
            gen_fake_distribute_file_json(distribute_file, world_size=world_size, local_rank=rank)


def test_me_de_train_dataset(batch_size=1, repeat_count=1, distribute_file='',
                             schema_file='./datasetSchema.json', data_files=[]):
    # apply repeat operations
    test_gen_fake_minddata()
    data_set = ds.StorageDataset(data_files, schema_file, distribution=distribute_file,
                                 columns_list=["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
                                               "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"])
    type_cast_op = deMap.TypeCastOp("DE_INT32")
    data_set = data_set.map(input_column_names="masked_lm_ids", operation=type_cast_op)
    data_set = data_set.map(input_column_names="masked_lm_positions", operation=type_cast_op)
    data_set = data_set.map(input_column_names="next_sentence_labels", operation=type_cast_op)
    data_set = data_set.map(input_column_names="segment_ids", operation=type_cast_op)
    data_set = data_set.map(input_column_names="input_mask", operation=type_cast_op)
    data_set = data_set.map(input_column_names="input_ids", operation=type_cast_op)
    data_set = data_set.repeat(repeat_count)

    # apply shuffle operations
    buffer_size = 640
    data_set = data_set.shuffle(buffer_size=buffer_size)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    data_set.channel_name = 'bert'
    return data_set


def test_bert_train():
    D.init()
    devid = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                        enable_hccl=True, device_id=devid)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, mirror_mean=True,
                                      device_num=D.get_group_size())

    version = os.getenv('VERSION', 'large')
    batch_size = int(os.getenv('BATCH_SIZE', '64'))
    epoch_size = 40

    config = get_config(version=version, batch_size=batch_size)
    netwithloss = BertNetworkWithLoss(config, True)

    # dataset
    dataset_dir = ""
    minddata_dir = ""
    schema_file = os.path.join(minddata_dir, 'datasetSchema.json')
    distribute_file = os.path.join(minddata_dir, 'minddata_json', 'distribution_{}.json'.format(D.get_rank()))
    files = os.listdir(dataset_dir)
    data_files = []
    for file_name in files:
        data_files.append(os.path.join(dataset_dir, file_name))
    dataset = test_me_de_train_dataset(batch_size, epoch_size, schema_file=schema_file,
                                       data_files=data_files, distribute_file=distribute_file)
    x = dataset.get_dataset_size()
    logger.info("dataset size is {}".format(x))

    optimizer = Lamb(netwithloss.trainable_params(), decay_steps=epoch_size * x, start_learning_rate=1e-4,
                     end_learning_rate=1e-8,
                     power=10.0, warmup_steps=3425, weight_decay=0.01, eps=1e-4)

    # checkpoint
    from mindspore.train.callback import CheckpointConfig
    from mindspore.train.callback import ModelCheckpoint
    ckpt_config = CheckpointConfig(save_checkpoint_steps=x, keep_checkpoint_max=2)
    ckpoint_cb = ModelCheckpoint(prefix='bert', directory=None, config=ckpt_config)

    # loss call back
    callback = LossCallBack()

    # loss scale
    scale_manager = DynamicLossScaleManager(2 ** 32, 2, 1000)
    update_cell = LossScaleUpdateCell(scale_manager)

    netwithgrads = BertTrainOneStepWithLossScaleCell(netwithloss, optimizer=optimizer, scale_update_cell=update_cell)

    model = Model(netwithgrads)
    model.train(epoch_size, dataset, callbacks=[callback, ckpoint_cb])
    D.release()
