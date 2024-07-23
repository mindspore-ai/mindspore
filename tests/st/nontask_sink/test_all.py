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
import os
from mindspore import context
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hccl_allreduce():
    """
    Feature: mpi run 8P case
    Description: mpi run 8P case
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    return_code = os.system("mpirun --allow-run-as-root -n 8 pytest -s test_allreduce.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_hccl_reduce():
    """
    Feature: mpi run 8P case of 'Reduce' communication operator.
    Description: mpi run 8P case of 'Reduce' communication operator.
    Expectation: success
    """
    return_code = os.system("mpirun --allow-run-as-root -n 8 pytest -s test_reduce.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_hccl_barrier():
    """
    Feature: mpi run 8P case of 'Barrier' communication operator.
    Description: mpi run 8P case of 'Barrier' communication operator.
    Expectation: success
    """
    return_code = os.system("mpirun --allow-run-as-root -n 8 pytest -s test_barrier.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hccl_allgather():
    """
    Feature: mpi run 8P case
    Description: mpi run 8P case
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    return_code = os.system("mpirun --allow-run-as-root -n 8 pytest -s test_allgather.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hccl_get_process_group_ranks_func_8p():
    """
    Feature: mpi run 8P case
    Description: mpi run 8P case
    Expectation: success
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    return_code = os.system("mpirun --allow-run-as-root -n 8 pytest -s test_get_process_group_ranks.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_hccl_batch_isend_irecv():
    """
    Feature: mpi run 8P case of 'BatchISendIRecv' communication operator.
    Description: mpi run 8P case of 'BatchISendIRecv' communication operator.
    Expectation: success
    """
    return_code = os.system("mpirun --allow-run-as-root -n 8 pytest -s test_batch_isend_irecv.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_hccl_all_to_all_single_with_output_shape():
    """
    Feature: mpi run 2P case of 'all_to_all_single_with_output_shape' communication operator.
    Description: mpi run 2P case of 'all_to_all_single_with_output_shape' communication operator.
    Expectation: success
    """
    return_code = os.system("mpirun --allow-run-as-root -n 2 pytest -s test_all_to_all_single_with_output_shape.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_hccl_broadcast():
    """
    Feature: mpi run 2P case of 'broadcast' communication operator.
    Description: mpi run 2P case of 'broadcast' communication operator.
    Expectation: success
    """
    return_code = os.system("mpirun --allow-run-as-root -n 2 pytest -s test_broadcast.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_hccl_gather_into_tensor():
    """
    Feature: mpi run 2P case of 'gather_into_tensor' communication operator.
    Description: mpi run 2P case of 'gather_into_tensor' communication operator.
    Expectation: success
    """
    return_code = os.system("mpirun --allow-run-as-root -n 2 pytest -s test_gather_into_tensor.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_hccl_scatter_tensor():
    """
    Feature: mpi run 2P case of 'scatter_tensor' communication operator.
    Description: mpi run 2P case of 'scatter_tensor' communication operator.
    Expectation: success
    """
    return_code = os.system("mpirun --allow-run-as-root -n 2 pytest -s "
                            "test_scatter_tensor.py::test_hccl_scatter_tensor_func_in_cell_2p")
    assert return_code == 0
    return_code = os.system("mpirun --allow-run-as-root -n 2 pytest -s "
                            "test_scatter_tensor.py::test_hccl_scatter_tensor_func_2p")
    assert return_code == 0
    return_code = os.system("mpirun --allow-run-as-root -n 4 pytest -s "
                            "test_scatter_tensor.py::test_scatter_tensor_two_groups")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_hccl_send_receive():
    """
    Feature: mpi run 2P case of 'send' and 'receive' communication operator.
    Description: mpi run 2P case of 'send' and 'receive' communication operator.
    Expectation: success
    """
    return_code = os.system("mpirun --allow-run-as-root -n 2 pytest -s test_send_receive.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_hccl_all_to_all_v():
    """
    Feature: mpi run 2P case of 'alltoallv' communication operator.
    Description: mpi run 2P case of 'alltoallv' communication operator.
    Expectation: success
    """
    return_code = os.system("mpirun --allow-run-as-root -n 2 pytest -s test_all_to_all_v.py")
    assert return_code == 0
