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
# ============================================================================
import os
import stat
import numpy as np
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import Parameter, Tensor, context
from mindspore.ops.auto_generate import WeightQuantBatchMatmul
from mindspore.common import dtype as mstype
from mindspore.nn.utils import no_init_parameters
from tests.mark_utils import arg_mark


class WeightQuantBatchMatmulNet(nn.Cell):
    """
    WeightQuantBatchMatmulNet.
    """

    def __init__(self, weight=None, transpose_x=False, transpose_weight=False, antiquant_group_size=0, strategy=None):
        super().__init__()
        np_int4_weight = np.ones([8, 16]).astype(np.float16)
        self.wqbmm = WeightQuantBatchMatmul(transpose_x, transpose_weight, antiquant_group_size)
        if weight is not None:
            self.weight = weight
        else:
            self.weight = Parameter(Tensor(np_int4_weight, dtype=mstype.qint4x2))

        self.scale = 0.1
        self.offset = 4.0

        self.antiquant_scale = Tensor([self.scale], dtype=mstype.float16)
        self.antiquant_offset = Tensor([-self.offset], dtype=mstype.float16)
        self.quant_scale = None
        self.quant_offset = None
        self.bias = None

        if strategy is not None:
            self.wqbmm.shard(strategy)

    def construct(self, x):
        out = self.wqbmm(x, self.weight, self.antiquant_scale, self.antiquant_offset, self.quant_scale,
                         self.quant_offset, self.bias)
        return out


def remove_ckpt(file_name):
    """remove ckpt."""
    if os.path.exists(file_name):
        os.chmod(file_name, stat.S_IWRITE)
        os.remove(file_name)


@arg_mark(plat_marks=['platform_ascend910b'],
        level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=20.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_save_load_sft_qint4x2_graph(mode):
    """
    Feature: save and load checkpoint
    Description: test ms.save_checkpoint and ms.load_checkpoint with qint4x2
    Expectation: success
    """
    context.set_context(mode=mode)
    net1 = WeightQuantBatchMatmulNet()
    ckpt_path = "checkpoint_1.safetensors"
    remove_ckpt(ckpt_path)
    ms.save_checkpoint(net1, ckpt_path, format="safetensors")

    net2 = WeightQuantBatchMatmulNet()
    output_param_dict = ms.load_checkpoint(ckpt_path, format="safetensors")
    assert output_param_dict["weight"].dtype == mstype.qint4x2
    ms.load_param_into_net(net2, output_param_dict)
    model = ms.Model(net2)
    predict_data = np.random.rand(8, 8).astype(np.float16)
    predict_result = model.predict(ms.Tensor(predict_data))
    net_result = net1(ms.Tensor(predict_data))
    assert np.array_equal(predict_result.asnumpy(), net_result.asnumpy()), "result not equal, please check"

    remove_ckpt(ckpt_path)


@arg_mark(plat_marks=['platform_ascend910b'],
        level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=20.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_save_load_sft_qint4x2_with_no_init_parameters(mode):
    """
    Feature: save and load checkpoint
    Description: test ms.save_checkpoint and ms.load_checkpoint with qint4x2 and no_init_parameters()
    Expectation: success
    """
    context.set_context(mode=mode)
    net1 = WeightQuantBatchMatmulNet()
    ckpt_path = "checkpoint_1.safetensors"
    remove_ckpt(ckpt_path)
    ms.save_checkpoint(net1, ckpt_path, format="safetensors")

    with no_init_parameters():
        net2 = WeightQuantBatchMatmulNet()
    output_param_dict = ms.load_checkpoint(ckpt_path, format="safetensors")
    assert output_param_dict["weight"].dtype == mstype.qint4x2
    ms.load_param_into_net(net2, output_param_dict)
    model = ms.Model(net2)
    predict_data = np.random.rand(8, 8).astype(np.float16)
    predict_result = model.predict(ms.Tensor(predict_data))
    net_result = net1(ms.Tensor(predict_data))
    assert np.array_equal(predict_result.asnumpy(), net_result.asnumpy()), "result not equal, please check"

    remove_ckpt(ckpt_path)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=18.0)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_save_and_load_sft_qint4x2_pynative(mode):
    """
    Feature: save and load checkpoint
    Description: test ms.save_checkpoint and ms.load_checkpoint with qint4x2 in pynative
    Expectation: success
    """
    context.set_context(mode=mode)
    net1 = WeightQuantBatchMatmulNet()
    ckpt_path = "checkpoint_1.safetensors"
    remove_ckpt(ckpt_path)
    ms.save_checkpoint(net1, ckpt_path, format="safetensors")

    net2 = WeightQuantBatchMatmulNet()
    output_param_dict = ms.load_checkpoint(ckpt_path, format="safetensors")
    assert output_param_dict["weight"].dtype == mstype.qint4x2
    ms.load_param_into_net(net2, output_param_dict)
    model = ms.Model(net2)
    predict_data = np.random.rand(8, 8).astype(np.float16)
    predict_result = model.predict(ms.Tensor(predict_data))
    net_result = net1(ms.Tensor(predict_data))
    assert np.array_equal(predict_result.asnumpy(), net_result.asnumpy()), "result not equal, please check"

    remove_ckpt(ckpt_path)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=20.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_save_and_load_sft_qint4x2_with_crc(mode):
    """
    Feature: save and load checkpoint
    Description: test ms.save_checkpoint and ms.load_checkpoint with qint4x2 and crc
    Expectation: success
    """
    context.set_context(mode=mode)
    net1 = WeightQuantBatchMatmulNet()
    ckpt_path = "checkpoint_1.safetensors"
    remove_ckpt(ckpt_path)
    ms.save_checkpoint(net1, ckpt_path, format="safetensors", crc_check=True)

    net2 = WeightQuantBatchMatmulNet()
    output_param_dict = ms.load_checkpoint(ckpt_path, format="safetensors", crc_check=True)
    assert output_param_dict["weight"].dtype == mstype.qint4x2
    ms.load_param_into_net(net2, output_param_dict)
    model = ms.Model(net2)
    predict_data = np.random.rand(8, 8).astype(np.float16)
    predict_result = model.predict(ms.Tensor(predict_data))
    net_result = net1(ms.Tensor(predict_data))
    assert np.array_equal(predict_result.asnumpy(), net_result.asnumpy()), "result not equal, please check"

    remove_ckpt(ckpt_path)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_safetensors_to_ckpt_with_qint4x2(mode):
    """
    Feature: safetensors_to_ckpt
    Description: test ms.safetensors_to_ckpt with qint4x2
    Expectation: success
    """
    context.set_context(mode=mode)
    net1 = WeightQuantBatchMatmulNet()
    ckpt_path = "./checkpoint_111.safetensors"
    remove_ckpt(ckpt_path)
    # save parameter(int4)
    ms.save_checkpoint(net1, ckpt_path, format="safetensors")

    ms.safetensors_to_ckpt(ckpt_path)

    output_param_dict = ms.load_checkpoint("./checkpoint_111.ckpt", format="ckpt")
    assert output_param_dict["weight"].dtype == mstype.qint4x2
    remove_ckpt(ckpt_path)
    remove_ckpt("./checkpoint_111.ckpt")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_ckpt_to_safetensors_with_qint4x2(mode):
    """
    Feature: ckpt_to_safetensors
    Description: test ms.ckpt_to_safetensors with qint4x2
    Expectation: success
    """
    context.set_context(mode=mode)
    net1 = WeightQuantBatchMatmulNet()
    ckpt_path = "./checkpoint_111.ckpt"
    remove_ckpt(ckpt_path)
    # save parameter(int4)
    ms.save_checkpoint(net1, ckpt_path, format="ckpt")

    ms.ckpt_to_safetensors(ckpt_path)

    output_param_dict = ms.load_checkpoint("./checkpoint_111.safetensors", format="safetensors")
    assert output_param_dict["weight"].dtype == mstype.qint4x2
    remove_ckpt(ckpt_path)
    remove_ckpt("./checkpoint_111.safetensors")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=20.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_save_and_load_ckpt_qint4x2(mode):
    """
    Feature: save and load checkpoint in format="ckpt"
    Description: test ms.save_checkpoint and ms.load_checkpoint with qint4x2 in ckpt
    Expectation: success
    """
    ckpt_format = 'ckpt'
    context.set_context(mode=mode)
    net1 = WeightQuantBatchMatmulNet()
    ckpt_path = "checkpoint_1.ckpt"
    remove_ckpt(ckpt_path)
    ms.save_checkpoint(net1, ckpt_path, format=ckpt_format)

    net2 = WeightQuantBatchMatmulNet()
    output_param_dict = ms.load_checkpoint(ckpt_path, format=ckpt_format)
    assert output_param_dict["weight"].dtype == mstype.qint4x2
    ms.load_param_into_net(net2, output_param_dict)
    model = ms.Model(net2)
    predict_data = np.random.rand(8, 8).astype(np.float16)
    predict_result = model.predict(ms.Tensor(predict_data))
    net_result = net1(ms.Tensor(predict_data))
    assert np.array_equal(predict_result.asnumpy(), net_result.asnumpy()), "result not equal, please check"

    remove_ckpt(ckpt_path)


@arg_mark(plat_marks=['platform_ascend910b'],
        level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_save_and_load_sft_qint4x2_with_strict_load(mode):
    """
    Feature: load checkpoint(strict_load=True)
    Description: test ms.load_checkpoint with qint4x2 and strict_load=True
    Expectation: runtimeerror
    """
    context.set_context(mode=mode)
    net1 = WeightQuantBatchMatmulNet()
    ckpt_path = "checkpoint_1.safetensors"
    remove_ckpt(ckpt_path)
    # save parameter(int4)
    ms.save_checkpoint(net1, ckpt_path, format="safetensors")

    int8_weight = np.ones([8, 16]).astype(np.float16)
    weight_int8 = Parameter(Tensor(int8_weight, dtype=mstype.int8))
    net2 = WeightQuantBatchMatmulNet(weight_int8)
    with pytest.raises(RuntimeError) as error_info:
        # load parameter(int4) into net(int8)
        ms.load_checkpoint(ckpt_path, net=net2, format="safetensors", strict_load=True)
    assert "For 'load_param_into_net', weight in the argument 'net' should have the same type" in str(error_info.value)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_save_unified_load_sft_qint4x2():
    '''
    Feature: train, unified, save load distribute sf with qint4x2
    Description: Test ms.save_checkpoint ms.unified_safetensors and ms.load_distributed_checkpoint
    Expectation: Run success
    '''
    train_ret = os.system(
        "msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 " \
        "--master_port=10820 --join=True --log_dir=./test_train_log " \
        "pytest -s parallel_qint4x2_net.py::test_train_model"
    )
    assert train_ret == 0
    unified_ret = os.system(
        "pytest -s parallel_qint4x2_net.py::test_unified_sf"
    )
    assert unified_ret == 0
    predict_ret = os.system(
        "msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 " \
        "--master_port=10820 --join=True --log_dir=./test_load_distributed_log " \
        "pytest -s parallel_qint4x2_net.py::test_load_distributed_predict_model"
    )
    assert predict_ret == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_save_unified_load_sft_qint4x2_with_remove_redundancy():
    '''
    Feature: train, unified, save load distribute sf with qint4x2
    Description: Test ms.save_checkpoint ms.unified_safetensors and ms.load_distributed_checkpoint
                 with remove_redundancy.
    Expectation: Run success
    '''
    train_ret = os.system(
        "msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 " \
        "--master_port=10820 --join=True --log_dir=./test_train_log " \
        "pytest -s parallel_qint4x2_net.py::test_train_model_remove_redundancy"
    )
    assert train_ret == 0
    unified_ret = os.system(
        "pytest -s parallel_qint4x2_net.py::test_unified_sf_remove_redundancy"
    )
    assert unified_ret == 0
    predict_ret = os.system(
        "msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 " \
        "--master_port=10820 --join=True --log_dir=./test_load_distributed_log " \
        "pytest -s parallel_qint4x2_net.py::test_load_distributed_predict_model"
    )
    assert predict_ret == 0
