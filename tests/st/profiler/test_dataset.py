# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Test dataset profiling."""
import os
import tempfile
import glob

import mindspore.dataset as ds
from mindspore.dataset import DSCallback
from mindspore import dtype as mstype
import mindspore.log as logger
import mindspore.dataset.transforms as transforms
import mindspore as ms
from mindspore.profiler import Profiler
from tests.mark_utils import arg_mark

MNIST_DIR = "/home/workspace/mindspore_dataset/mnist/"
CIFAR10_DIR = "/home/workspace/mindspore_dataset/cifar-10-batches-bin/"


def create_dict_iterator(datasets):
    """create_dict_iterator"""
    count = 0
    for _ in datasets.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1


class PrintInfo(DSCallback):
    """PrintInfo"""

    @staticmethod
    def ds_begin(ds_run_context):
        """ds_begin"""
        logger.info("callback: start dataset pipeline", ds_run_context.cur_epoch_num)

    @staticmethod
    def ds_epoch_begin(ds_run_context):
        """ds_epoch_begin"""
        logger.info("callback: epoch begin, we are in epoch", ds_run_context.cur_epoch_num)

    @staticmethod
    def ds_epoch_end(ds_run_context):
        """ds_epoch_end"""
        logger.info("callback: epoch end, we are in epoch", ds_run_context.cur_epoch_num)

    @staticmethod
    def ds_step_begin(ds_run_context):
        """ds_step_begin"""
        logger.info("callback: step start, we are in epoch", ds_run_context.cur_step_num)

    @staticmethod
    def ds_step_end(ds_run_context):
        """ds_step_end"""
        logger.info("callback: step end, we are in epoch", ds_run_context.cur_step_num)


def add_one_by_epoch(batchinfo):
    """add_one_by_epoch"""
    return batchinfo.get_epoch_num() + 1


def other_method_dataset():
    """create other_method dataset"""
    path_base = os.path.split(os.path.realpath(__file__))[0]
    data = []
    for d in range(10):
        data.append(d)
    dataset = ds.GeneratorDataset(data, "column1")
    dataset = dataset.batch(batch_size=add_one_by_epoch)
    create_dict_iterator(dataset)

    dataset = ds.GeneratorDataset([1, 2], "col1", shuffle=False, num_parallel_workers=1)
    dataset = dataset.map(operations=lambda x: x, callbacks=PrintInfo())
    create_dict_iterator(dataset)

    schema = ds.Schema()
    schema.add_column(name='col1', de_type=mstype.int64, shape=[2])
    columns1 = [{'name': 'image', 'type': 'int8', 'shape': [3, 3]},
                {'name': 'label', 'type': 'int8', 'shape': [1]}]
    schema.parse_columns(columns1)

    pipeline1 = ds.MnistDataset(MNIST_DIR, num_samples=10)
    pipeline2 = ds.Cifar10Dataset(CIFAR10_DIR, num_samples=10)
    ds.compare(pipeline1, pipeline2)

    dataset = ds.MnistDataset(MNIST_DIR, num_samples=10)
    one_hot_encode = transforms.OneHot(10)
    dataset = dataset.map(operations=one_hot_encode, input_columns="label")
    dataset = dataset.batch(batch_size=10, drop_remainder=True)
    ds.serialize(dataset, json_filepath=os.path.join(path_base, "mnist_dataset_pipeline.json"))
    ds.show(dataset)
    serialized_data = ds.serialize(dataset)
    ds.deserialize(input_dict=serialized_data)
    return dataset


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_ascend_dataset_profiler():
    """
    Feature: Test the dataset profiling.
    Description: Traverse the dataset data, perform data preprocessing, and then verify the collected profiling data.
    Expectation: No dataset_iterator_profiling file generated.
    """
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory() as tmpdir:
        profiler = Profiler(output_path=tmpdir)
        other_method_dataset()
        profiler.analyse()
        assert len(glob.glob(f"{tmpdir}/profiler*/dataset_iterator_profiling_*.txt")) == 1
