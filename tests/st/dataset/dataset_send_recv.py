# Copyright 2025 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
Test dataset with send and recv
"""

from mindspore import Tensor
from mindspore import log as logger
from mindspore.common import dtype as mstype
from mindspore.dataset import GeneratorDataset
from mindspore.mint.distributed import init_process_group
from mindspore.mint.distributed import send, recv, get_rank

import numpy as np
import pytest

def create_dataset(this_rank):
    # Random-accessible object as input source
    class RandomAccessDataset:
        def __init__(self, this_rank):
            self._data = np.ones((5, 2))
            self._label = np.zeros((5, 1), dtype=np.float32)
            self.this_rank = this_rank
        def __getitem__(self, index):
            return np.ones((index + 1, 1 + index + self.this_rank * 5), dtype=np.int32), \
                   self._label[index] + index + self.this_rank * 5
        def __len__(self):
            return len(self._data)

    loader = RandomAccessDataset(this_rank)
    dataset = GeneratorDataset(source=loader, column_names=["data", "label"], shuffle=False)
    return dataset


def check_dataset_send_recv_1_to_1(dataset, data=None, src=None, dst=None):
    this_rank = get_rank()
    if this_rank == src:
        if data is not None:
            dataset.send(data, dst)
        else:
            dataset.send(dst=dst)

    if this_rank == dst:
        if not isinstance(data, list):
            if data.dtype in [mstype.complex64, mstype.cfloat, mstype.complex128, mstype.cdouble]:
                # not support, not need to recv
                return
            result = dataset.recv(src)
            assert result.shape == data.shape
            assert result.dtype == data.dtype
            assert (result.asnumpy() == data.asnumpy()).all()
        else:
            result = dataset.recv(src)
            for index, expect in enumerate(data):
                assert result[index].shape == expect.shape
                assert result[index].dtype == expect.dtype
                assert (result[index].asnumpy() == expect.asnumpy()).all()

def dataset_send_recv_1_to_1():
    """
    Feature: Dataset with send & recv
    Description: send to recv is 1:1
    Expectation: Success
    """
    logger.info(">>>> testcase dataset_send_recv_1_to_1 >>>>")

    init_process_group()
    this_rank = get_rank()
    logger.info(f"this rank: {this_rank}")

    # basic send & recv by mint
    if this_rank == 0:
        input_ = Tensor(np.ones([2, 8]).astype(np.float32))
        send(input_, 1)
    if this_rank == 1:
        x = Tensor(np.zeros([2, 8]).astype(np.float32))
        _ = recv(x, src=0)

    dataset = create_dataset(this_rank)
    for item in dataset:
        logger.info(item)

    np.random.seed(1234)

    # dataset send & recv
    # test all dtype
    input_tensor1 = Tensor(np.random.random_sample([2,]), dtype=mstype.bool)
    input_tensor2 = Tensor(np.random.random_sample([3,]) * 256, dtype=mstype.int8)
    input_tensor3 = Tensor(np.random.random_sample([4,]) * 256, dtype=mstype.int16)
    input_tensor4 = Tensor(np.random.random_sample([5,]) * 256, dtype=mstype.short)
    input_tensor5 = Tensor(np.random.random_sample([6, 8]) * 256, dtype=mstype.int32)
    input_tensor6 = Tensor(np.random.random_sample([7, 8]) * 256, dtype=mstype.int)
    input_tensor7 = Tensor(np.random.random_sample([8, 8]) * 256, dtype=mstype.int64)
    input_tensor8 = Tensor(np.random.random_sample([9, 8]) * 256, dtype=mstype.long)
    input_tensor9 = Tensor(np.random.random_sample([10, 8]) * 256, dtype=mstype.uint8)
    input_tensor10 = Tensor(np.random.random_sample([11, 8, 4]) * 256, dtype=mstype.uint16)
    input_tensor11 = Tensor(np.random.random_sample([12, 8, 4]) * 256, dtype=mstype.uint32)
    input_tensor12 = Tensor(np.random.random_sample([13, 8, 4]) * 256, dtype=mstype.uint64)
    input_tensor13 = Tensor(np.random.random_sample([14, 8, 4]) * 256, dtype=mstype.float16)
    input_tensor14 = Tensor(np.random.random_sample([15, 8, 4]) * 256, dtype=mstype.half)
    input_tensor15 = Tensor(np.random.random_sample([16, 8, 4, 2]) * 256, dtype=mstype.float32)
    input_tensor16 = Tensor(np.random.random_sample([17, 8, 4, 2]) * 256, dtype=mstype.float)
    input_tensor17 = Tensor(np.random.random_sample([18, 8, 4, 2]) * 256, dtype=mstype.float64)
    input_tensor18 = Tensor(np.random.random_sample([19, 8, 4, 2]) * 256, dtype=mstype.double)
    input_tensor19 = Tensor(np.random.random_sample([20, 8, 4, 2]) * 256, dtype=mstype.bfloat16)
    check_dataset_send_recv_1_to_1(dataset, input_tensor1, 1, 0)
    check_dataset_send_recv_1_to_1(dataset, input_tensor2, 1, 0)
    check_dataset_send_recv_1_to_1(dataset, input_tensor3, 1, 0)
    check_dataset_send_recv_1_to_1(dataset, input_tensor4, 1, 0)
    check_dataset_send_recv_1_to_1(dataset, input_tensor5, 1, 0)
    check_dataset_send_recv_1_to_1(dataset, input_tensor6, 1, 0)
    check_dataset_send_recv_1_to_1(dataset, input_tensor7, 1, 0)
    check_dataset_send_recv_1_to_1(dataset, input_tensor8, 1, 0)
    check_dataset_send_recv_1_to_1(dataset, input_tensor9, 1, 0)
    check_dataset_send_recv_1_to_1(dataset, input_tensor10, 1, 0)
    check_dataset_send_recv_1_to_1(dataset, input_tensor11, 1, 0)
    check_dataset_send_recv_1_to_1(dataset, input_tensor12, 1, 0)
    check_dataset_send_recv_1_to_1(dataset, input_tensor13, 1, 0)
    check_dataset_send_recv_1_to_1(dataset, input_tensor14, 1, 0)
    check_dataset_send_recv_1_to_1(dataset, input_tensor15, 1, 0)
    check_dataset_send_recv_1_to_1(dataset, input_tensor16, 1, 0)
    check_dataset_send_recv_1_to_1(dataset, input_tensor17, 1, 0)
    check_dataset_send_recv_1_to_1(dataset, input_tensor18, 1, 0)
    check_dataset_send_recv_1_to_1(dataset, input_tensor19, 1, 0)   # 910A not support

    # test input tensor is list
    check_dataset_send_recv_1_to_1(dataset, [input_tensor1, input_tensor2, input_tensor3, input_tensor4, input_tensor5,
                                             input_tensor5, input_tensor7, input_tensor8, input_tensor9, input_tensor10,
                                             input_tensor11, input_tensor12, input_tensor13, input_tensor14,
                                             input_tensor15, input_tensor16, input_tensor17, input_tensor18], 1, 0)

    # send data from rank 1 to rank 0
    for index in range(dataset.get_dataset_size()):
        if this_rank == 1:
            dataset.send(dst=0)

        if this_rank == 0:
            # get data from current dataset
            output_tensors0 = dataset.recv(src=0)
            assert len(output_tensors0) == 2
            data0 = Tensor(np.ones((index + 1, 1 + index + this_rank * 5)), dtype=mstype.int32)
            label0 = Tensor(np.zeros((1,)) + index + this_rank * 5, dtype=mstype.float32)
            assert data0.shape == output_tensors0[0].shape
            assert data0.dtype == output_tensors0[0].dtype
            assert (data0.asnumpy() == output_tensors0[0].asnumpy()).all()
            assert label0.shape == output_tensors0[1].shape
            assert label0.dtype == output_tensors0[1].dtype
            assert (label0.asnumpy() == output_tensors0[1].asnumpy()).all()

            # get data from rank 1
            output_tensors1 = dataset.recv(src=1)
            assert len(output_tensors1) == 2
            data1 = Tensor(np.ones((index + 1, 1 + index + 1 * 5)), dtype=mstype.int32)
            label1 = Tensor(np.zeros((1,)) + index + 1 * 5, dtype=mstype.float32)
            assert data1.shape == output_tensors1[0].shape
            assert data1.dtype == output_tensors1[0].dtype
            assert (data1.asnumpy() == output_tensors1[0].asnumpy()).all()
            assert label1.shape == output_tensors1[1].shape
            assert label1.dtype == output_tensors1[1].dtype
            assert (label1.asnumpy() == output_tensors1[1].asnumpy()).all()


def dataset_send_recv_1_to_n():
    """
    Feature: Dataset with send & recv
    Description: send to recv is 1:n
    Expectation: Success
    """
    logger.info(">>>> testcase dataset_send_recv_1_to_n >>>>")

    init_process_group()
    this_rank = get_rank()
    logger.info(f"this rank: {this_rank}")

    # basic send & recv by mint
    if this_rank == 0:
        input_ = Tensor(np.ones([2, 8]).astype(np.float32))
        send(input_, 1)
    if this_rank == 1:
        x = Tensor(np.zeros([2, 8]).astype(np.float32))
        _ = recv(x, src=0)

    dataset = create_dataset(this_rank)
    for item in dataset:
        logger.info(item)

    np.random.seed(1234)

    # rank: 0                        1        2        3        4        5        6        7
    # data = dataset.recv(0)
    # dataset.send(data[1], 7)                                                             dataset.recv(0)
    # data = dataset.recv(1)         dataset.send(0)
    # dataset.send(data[1], 7)                                                             dataset.recv(0)
    # data = dataset.recv(2)                  dataset.send(0)
    # dataset.send(data[1], 7)                                                             dataset.recv(0)
    # data = dataset.recv(3)                           dataset.send(0)
    # dataset.send(data[1], 7)                                                             dataset.recv(0)
    # data = dataset.recv(4)                                    dataset.send(0)
    # dataset.send(data[1], 7)                                                             dataset.recv(0)
    # data = dataset.recv(5)                                             dataset.send(0)
    # dataset.send(data[1], 7)                                                             dataset.recv(0)
    # data = dataset.recv(6)                                                      dataset.send(0)
    # dataset.send(data[1], 7)                                                             dataset.recv(0)
    # data = dataset.recv(7)                                                               dataset.send(0)
    # dataset.send(data[1], 7)                                                             dataset.recv(0)
    # data = dataset.recv(0)
    # dataset.send(data[1], 7)                                                             dataset.recv(0)
    # data = dataset.recv(1)         dataset.send(0)
    # dataset.send(data[1], 7)                                                             dataset.recv(0)
    #                                        ......
    logger.info("==== test1 ====")
    for _ in range(5):  # 5 epoch
        for index in range(dataset.get_dataset_size()):
            if this_rank == 0:
                for src_rank in range(8):
                    # get data fom the dataset
                    output_tensors0 = dataset.recv(src=src_rank)
                    assert len(output_tensors0) == 2
                    data0 = Tensor(np.ones((index + 1, 1 + index + src_rank * 5)), dtype=mstype.int32)
                    label0 = Tensor(np.zeros((1,)) + index + src_rank * 5, dtype=mstype.float32)
                    assert data0.shape == output_tensors0[0].shape
                    assert data0.dtype == output_tensors0[0].dtype
                    assert (data0.asnumpy() == output_tensors0[0].asnumpy()).all()
                    assert label0.shape == output_tensors0[1].shape
                    assert label0.dtype == output_tensors0[1].dtype
                    assert (label0.asnumpy() == output_tensors0[1].asnumpy()).all()

                    dataset.send(tensor=label0, dst=7)

            if this_rank == 7:
                for src_rank in range(7):
                    output_label = dataset.recv(src=0)  # recv label
                    label0 = Tensor(np.zeros((1,)) + index + src_rank * 5, dtype=mstype.float32)
                    assert (label0.asnumpy() == output_label.asnumpy()).all()
                dataset.send(dst=0)
                output_label = dataset.recv(src=0)
                label0 = Tensor(np.zeros((1,)) + index + this_rank * 5, dtype=mstype.float32)
                assert (label0.asnumpy() == output_label.asnumpy()).all()

            if this_rank in [1,2,3,4,5,6]:
                dataset.send(dst=0)

    # rank: 0                                    1        2        3        4        5        6        7
    # data = dataset.recv([0,1,2,3,4,5,6,7])     dataset.send(0)                 ......                dataset.send(0)
    # dataset.send(labels, 7)                                                                          dataset.recv(0)
    # data = dataset.recv([0,1,2,3,4,5,6,7])     dataset.send(0)                 ......                dataset.send(0)
    # dataset.send(labels, 7)                                                                          dataset.recv(0)
    logger.info("==== test2 ====")
    for _ in range(5):  # 5 epoch
        for index in range(dataset.get_dataset_size()):
            if this_rank == 0:
                # data is: [[t1, t2], [t3, t4], [t5, t6], ...]
                data = dataset.recv([0, 1, 2, 3, 4, 5, 6, 7])
                assert len(data) == 8
                extract_data = []  # [t1, t3, t5, t7, ...]
                extract_label = []  # [t2, t4, t6, t8, ...]
                for item in data:
                    assert len(item) == 2
                    extract_data.append(item[0])
                    extract_label.append(item[1])

                src_rank = 0
                for a, b in zip(extract_data, extract_label):
                    data0 = Tensor(np.ones((index + 1, 1 + index + src_rank * 5)), dtype=mstype.int32)
                    label0 = Tensor(np.zeros((1,)) + index + src_rank * 5, dtype=mstype.float32)
                    assert data0.shape == a.shape
                    assert data0.dtype == a.dtype
                    assert (data0.asnumpy() == a.asnumpy()).all()
                    assert label0.shape == b.shape
                    assert label0.dtype == b.dtype
                    assert (label0.asnumpy() == b.asnumpy()).all()
                    src_rank += 1

                dataset.send(tensor=extract_label, dst=7)

            if this_rank == 7:
                dataset.send(dst=0)
                all_labels = dataset.recv(src=0)

                for src_rank, b in enumerate(all_labels):
                    label0 = Tensor(np.zeros((1,)) + index + src_rank * 5, dtype=mstype.float32)
                    assert label0.shape == b.shape
                    assert label0.dtype == b.dtype
                    assert (label0.asnumpy() == b.asnumpy()).all()

            if this_rank in [1,2,3,4,5,6]:
                dataset.send(dst=0)

    ## send & recv data and label separately
    # rank: 0                    1      2      3      4      5      6                           7
    # data = dataset.recv(0)
    # dataset.send(data[1], 7)                                                                  dataset.recv(0)
    #                            data = dataset.recv(1)
    # data = dataset.recv(1)     dataset.send(data[0], 0)
    #                            dataset.send(data[1], 7)                                       dataset.recv(1)
    #                                   data = dataset.recv(2)
    # data = dataset.recv(2)            dataset.send(data[0], 0)
    #                                   dataset.send(data[1], 7)                                dataset.recv(2)
    #                                          data = dataset.recv(3)
    # data = dataset.recv(3)                   dataset.send(data[0], 0)
    #                                          dataset.send(data[1], 7)                         dataset.recv(3)
    #                                                 data = dataset.recv(4)
    # data = dataset.recv(4)                          dataset.send(data[0], 0)
    #                                                 dataset.send(data[1], 7)                  dataset.recv(4)
    #                                                        data = dataset.recv(5)
    # data = dataset.recv(5)                                 dataset.send(data[0], 0)
    #                                                        dataset.send(data[1], 7)           dataset.recv(5)
    #                                                               data = dataset.recv(6)
    # data = dataset.recv(6)                                        dataset.send(data[0], 0)
    #                                                               dataset.send(data[1], 7)    dataset.recv(6)
    #                                                                                           data = dataset.recv(7)
    # data = dataset.recv(7)                                                                    dataset.send(data[0], 0)
    # data = dataset.recv(0)
    # dataset.send(data[1], 7)                                                                  dataset.recv(0)
    #                            data = dataset.recv(1)
    # data = dataset.recv(1)     dataset.send(data[0], 0)
    #                            dataset.send(data[1], 7)                                       dataset.recv(1)
    #                                            ......
    logger.info("==== test3 ====")
    for _ in range(5):  # 5 epoch
        for index in range(dataset.get_dataset_size()):
            if this_rank == 0:
                for src_rank in range(8):
                    if src_rank == this_rank:
                        data = dataset.recv(src_rank)
                        dataset.send(data[1], 7)
                    else:
                        data = dataset.recv(src_rank)
                        data0 = Tensor(np.ones((index + 1, 1 + index + src_rank * 5)), dtype=mstype.int32)
                        assert data0.shape == data.shape
                        assert data0.dtype == data.dtype
                        assert (data0.asnumpy() == data.asnumpy()).all()

            if this_rank == 7:
                for src_rank in range(8):
                    if src_rank == this_rank:
                        data_label = dataset.recv(src_rank)
                        dataset.send(data_label[0], 0)
                    else:
                        label = dataset.recv(src_rank)
                        label0 = Tensor(np.zeros((1,)) + index + src_rank * 5, dtype=mstype.float32)
                        assert label0.shape == label.shape
                        assert label0.dtype == label.dtype
                        assert (label0.asnumpy() == label.asnumpy()).all()

            if this_rank in [1,2,3,4,5,6]:
                data_label = dataset.recv(this_rank)
                dataset.send(data_label[0], 0)
                dataset.send(data_label[1], 7)

    ## send & recv data and label separately
    # rank: 0                                1      2      3      4      5      6        7
    # data_label = dataset.recv(0)           data_label = dataset.recv(1)      ......    data_label = dataset.recv(7)
    # data1_7 = dataset.recv(1,2,3,4,5,6,7)  dataset.send(data_label[0], 0)    ......    dataset.send(data_label[0], 0)
    # dataset.send(data_label[1], 7)         dataset.send(data_label[1], 7)    ......    dataset.recv([0,1,2,3,4,5,6])
    # data_label = dataset.recv(0)           data_label = dataset.recv(1)      ......    data_label = dataset.recv(7)
    # data1_7 = dataset.recv(1,2,3,4,5,6,7)  dataset.send(data_label[0], 0)    ......    dataset.send(data_label[0], 0)
    # dataset.send(data_label[1], 7)         dataset.send(data_label[1], 7)    ......    dataset.recv([0,1,2,3,4,5,6])
    #                                               ......
    logger.info("==== test4 ====")
    for _ in range(5):  # 5 epoch
        for index in range(dataset.get_dataset_size()):
            data_label = dataset.recv(this_rank)
            if this_rank == 0:
                data_list = dataset.recv([1,2,3,4,5,6,7])  # [t2, t3, t4, t5, ...]
                datas_collect = []
                datas_collect.append(data_label[0])
                datas_collect.extend(data_list)

                for src_rank, data in enumerate(datas_collect):
                    data0 = Tensor(np.ones((index + 1, 1 + index + src_rank * 5)), dtype=mstype.int32)
                    assert data0.shape == data.shape
                    assert data0.dtype == data.dtype
                    assert (data0.asnumpy() == data.asnumpy()).all()

                dataset.send(data_label[1], 7)

            if this_rank == 7:
                dataset.send(data_label[0], 0)
                labels = dataset.recv([0,1,2,3,4,5,6])  # [t10, t11, t12, t13, ...]
                labels_collect = []
                labels_collect.extend(labels)
                labels_collect.append(data_label[1])

                for src_rank, label in enumerate(labels_collect):
                    label0 = Tensor(np.zeros((1,)) + index + src_rank * 5, dtype=mstype.float32)
                    assert label0.shape == label.shape
                    assert label0.dtype == label.dtype
                    assert (label0.asnumpy() == label.asnumpy()).all()

            if this_rank in [1,2,3,4,5,6]:
                dataset.send(data_label[0], 0)
                dataset.send(data_label[1], 7)


def dataset_send_recv_n_to_1():
    """
    Feature: Dataset with send & recv
    Description: send to recv is n:1
    Expectation: Success
    """
    logger.info(">>>> testcase dataset_send_recv_n_to_1 >>>>")

    init_process_group()
    this_rank = get_rank()
    logger.info(f"this rank: {this_rank}")

    # basic send & recv by mint
    if this_rank == 0:
        input_ = Tensor(np.ones([2, 8]).astype(np.float32))
        send(input_, 1)
    if this_rank == 1:
        x = Tensor(np.zeros([2, 8]).astype(np.float32))
        _ = recv(x, src=0)

    dataset = create_dataset(this_rank)
    for item in dataset:
        logger.info(item)

    np.random.seed(1234)

    # send data from rank:0 to rank:1,2,3,4,5,6,7
    input_tensor1 = Tensor(np.random.random_sample([13, 8, 20]) * 256, dtype=mstype.uint64)
    input_tensor2 = Tensor(np.random.random_sample([4, 10, 16]), dtype=mstype.float16)
    if this_rank == 0:
        dataset.send(tensor=[input_tensor1, input_tensor2], dst=[1,2,3,4,5,6,7])
    if this_rank in [1, 2, 3, 4, 5, 6, 7]:
        data = dataset.recv(0)
        assert len(data) == 2
        assert input_tensor1.shape == data[0].shape
        assert input_tensor1.dtype == data[0].dtype
        assert (input_tensor1.asnumpy() == data[0].asnumpy()).all()
        assert input_tensor2.shape == data[1].shape
        assert input_tensor2.dtype == data[1].dtype
        assert (input_tensor2.asnumpy() == data[1].asnumpy()).all()


    # send data from dataset from rank:0 to rank:1,2,3,4,5,6,7
    for _ in range(5):  # 5 epoch
        for index in range(dataset.get_dataset_size()):
            if this_rank == 0:
                dataset.send(dst=[1,2,3,4,5,6,7])
            if this_rank in [1, 2, 3, 4, 5, 6, 7]:
                data = dataset.recv(0)
                data0 = Tensor(np.ones((index + 1, 1 + index + 0 * 5)), dtype=mstype.int32)
                label0 = Tensor(np.zeros((1,)) + index + 0 * 5, dtype=mstype.float32)
                assert data0.shape == data[0].shape
                assert data0.dtype == data[0].dtype
                assert (data0.asnumpy() == data[0].asnumpy()).all()
                assert label0.shape == data[1].shape
                assert label0.dtype == data[1].dtype
                assert (label0.asnumpy() == data[1].asnumpy()).all()


def dataset_send_recv_n_to_n():
    """
    Feature: Dataset with send & recv
    Description: send to recv is n:n
    Expectation: Success
    """
    logger.info(">>>> testcase dataset_send_recv_n_to_n >>>>")

    init_process_group()
    this_rank = get_rank()
    logger.info(f"this rank: {this_rank}")

    # basic send & recv by mint
    if this_rank == 0:
        input_ = Tensor(np.ones([2, 8]).astype(np.float32))
        send(input_, 1)
    if this_rank == 1:
        x = Tensor(np.zeros([2, 8]).astype(np.float32))
        _ = recv(x, src=0)

    dataset = create_dataset(this_rank)
    for item in dataset:
        logger.info(item)

    np.random.seed(1234)

    # send data from rank:0 to rank:1,2,3,4,5,6,7
    input_tensor1 = Tensor(np.random.random_sample([13, 8, 20]) * 256, dtype=mstype.int32)
    if this_rank in [0, 1, 2, 3]:
        dataset.send(tensor=input_tensor1 * this_rank, dst=[4,5,6,7])
    if this_rank in [4, 5, 6, 7]:
        data = dataset.recv([0, 1, 2, 3])
        assert len(data) == 4
        for index, item in enumerate(data):
            assert input_tensor1.shape == item.shape
            assert input_tensor1.dtype == item.dtype
            assert ((input_tensor1 * index).asnumpy() == item.asnumpy()).all()


    # send data from dataset from rank:0 to rank:1,2,3,4,5,6,7
    for _ in range(5):  # 5 epoch
        for index in range(dataset.get_dataset_size()):
            if this_rank in [0, 1, 2, 3]:
                dataset.send(dst=[4,5,6,7])
            if this_rank in [4, 5, 6, 7]:
                data = dataset.recv([0, 1, 2, 3])
                assert len(data) == 4
                for src_rank, item in enumerate(data):
                    data0 = Tensor(np.ones((index + 1, 1 + index + src_rank * 5)), dtype=mstype.int32)
                    label0 = Tensor(np.zeros((1,)) + index + src_rank * 5, dtype=mstype.float32)
                    assert data0.shape == item[0].shape
                    assert data0.dtype == item[0].dtype
                    assert (data0.asnumpy() == item[0].asnumpy()).all()
                    assert label0.shape == item[1].shape
                    assert label0.dtype == item[1].dtype
                    assert (label0.asnumpy() == item[1].asnumpy()).all()

def dataset_send_recv_exception():
    """
    Feature: Dataset with send & recv with exception
    Description: send to recv when exception
    Expectation: Success
    """
    logger.info(">>>> testcase dataset_send_recv_exception>>>>")

    init_process_group()
    this_rank = get_rank()
    logger.info(f"this rank: {this_rank}")

    # basic send & recv by mint
    if this_rank == 0:
        input_ = Tensor(np.ones([2, 8]).astype(np.float32))
        send(input_, 1)
    if this_rank == 1:
        x = Tensor(np.zeros([2, 8]).astype(np.float32))
        _ = recv(x, src=0)

    dataset = create_dataset(this_rank)
    for item in dataset:
        logger.info(item)

    np.random.seed(1234)

    input_tensor1 = Tensor(np.random.random_sample([2,]), dtype=mstype.bool)
    input_tensor9 = Tensor(np.random.random_sample([10, 8]) * 256, dtype=mstype.uint8)
    input_tensor20 = Tensor([1.5 + 2.5j, 3.0 + 4.0j], mstype.complex64)
    input_tensor21 = Tensor([1.5 + 3.5j, 3.0 + 5.0j], mstype.cfloat)
    input_tensor22 = Tensor([1.5 + 4.5j, 3.0 + 6.0j], mstype.complex128)
    input_tensor23 = Tensor([1.5 + 5.5j, 3.0 + 7.0j], mstype.cdouble)
    with pytest.raises(RuntimeError) as info:
        dataset.send(input_tensor20, 0)
    assert "is not supported to send" in str(info)

    with pytest.raises(RuntimeError) as info:
        dataset.send(input_tensor21, 0)
    assert "is not supported to send" in str(info)

    with pytest.raises(RuntimeError) as info:
        dataset.send(input_tensor22, 0)
    assert "is not supported to send" in str(info)

    with pytest.raises(RuntimeError) as info:
        dataset.send(input_tensor23, 0)
    assert "is not supported to send" in str(info)

    # invalid parameter
    with pytest.raises(TypeError) as info:
        x = np.zeros((2, 2))
        dataset.send(x)
    assert "is not of type" in str(info)

    with pytest.raises(TypeError) as info:
        x = [np.zeros((2, 2))]
        dataset.send(x)
    assert "is not of type" in str(info)

    with pytest.raises(TypeError) as info:
        dataset.send(input_tensor1, "1")
    assert "is not of type" in str(info)

    with pytest.raises(ValueError) as info:
        dataset.send(input_tensor1, -1)
    assert "is not within the required interval of" in str(info)

    with pytest.raises(TypeError) as info:
        dataset.send(input_tensor1, ["1"])
    assert "is not of type" in str(info)

    with pytest.raises(TypeError) as info:
        dataset.send(input_tensor1, 1, 100)
    assert "is not of type" in str(info)

    # send to self
    with pytest.raises(ValueError) as info:
        dataset.send(input_tensor9, get_rank())
    assert "Invalid destination rank: destination rank should not be the same as the rank of the current process" \
        in str(info)

    # invalid group
    if this_rank == 0:
        with pytest.raises(KeyError) as info:
            dataset.send(input_tensor9, 1, "hccl_test")
        assert "is not found in GROUP_RANKS" in str(info)

    # invalid data with str
    def map_str(data):
        return data, np.array("abc")
    dataset = dataset.map(map_str, input_columns=["data"], output_columns=["data", "str"])
    with pytest.raises(RuntimeError) as info:
        dataset.send()
    assert "is not Tensor which is not supported to send" in str(info)


if __name__ == '__main__':
    dataset_send_recv_1_to_1()
    dataset_send_recv_1_to_n()
    dataset_send_recv_n_to_1()
    dataset_send_recv_n_to_n()
    dataset_send_recv_exception()
