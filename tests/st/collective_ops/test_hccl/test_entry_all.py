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
The tests of mindspore, used to test communication ops.
"""

import os
import subprocess
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
    return_code = os.system("msrun --worker_num=8 --local_worker_num=8 --master_port=12345 pytest -s test_allreduce.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
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
def test_get_comm_name_create_group_with_options_func_8p():
    """
    Feature: msrun 8P case
    Description: msrun 8P case
    Expectation: success
    """
    os.environ['GLOG_v'] = str(2)
    return_code = os.system(
        "export HCCL_BUFFSIZE=300; msrun --worker_num=8 --local_worker_num=8 --master_port=10969 --join=True "\
        "--log_dir=get_comm_name_create_group_options pytest -s test_get_comm_name_create_group.py"
    )
    assert return_code == 0
    result = subprocess.getoutput("grep -rn 'HcclCommInitRootInfoConfig for group_buffersize' "
                                  "./get_comm_name_create_group_options/worker_0.log")
    assert result.find("HcclCommInitRootInfoConfig for group_buffersize_default, hcclBufferSize is 300 MB") != -1
    assert result.find("HcclCommInitRootInfoConfig for group_buffersize_400, hcclBufferSize is 400 MB") != -1
    assert result.find("HcclCommInitRootInfoConfig for group_buffersize_100, hcclBufferSize is 100 MB") != -1


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
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


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_allgather_v():
    """
    Feature: mpi run 2P case of 'allgather_v' communication operator.
    Description: mpi run 2P case of 'allgather_v' communication operator.
    Expectation: success
    """
    return_code = os.system("mpirun --allow-run-as-root -n 2 pytest -s test_allgather_v.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_allgatherv_grad():
    """
    Feature: mpi run 2P case of allgather_v grad communication operator.
    Description: mpi run 2P case of allgather_v grad communication operator.
    Expectation: success
    """
    return_code = os.system("msrun --worker_num=2 --local_worker_num=2 --join=True pytest -sv  test_allgatherv_grad.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_dynamic_shape():
    """
    Feature: mpi run 2P case of communication dynamic shape operator.
    Description: mpi run 2P case of communication dynamic shape operator.
    Expectation: success
    """
    return_code = os.system("msrun --worker_num=8 --local_worker_num=8 --join=True pytest -sv  test_dynamic_shape.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_reduce_scatter_tensor_v():
    """
    Feature: mpi run 2P case of 'reduce_scatter_tensor_v' communication operator.
    Description: mpi run 2P case of 'reduce_scatter_tensor_v' communication operator.
    Expectation: success
    """
    return_code = os.system("mpirun --allow-run-as-root -n 2 pytest -s test_reduce_scatter_tensor_v.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_reduce_scatter_tensor_with_diff_reduce_type():
    """
    Feature: test reduce_scatter with different reducing type.
    Description: test reduce_scatter with different reducing type.
    Expectation: success
    """
    return_code = os.system("msrun --worker_num=8 --local_worker_num=8 --join=True \
                             pytest -s test_reduce_scatter_tensor.py::test_hccl_reduce_scatter_diff_reduce_type")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_all_to_all_v_c():
    """
    Feature: mpi run 2P case of 'all_to_all_v_c' communication operator.
    Description: mpi run 2P case of 'all_to_all_v_c' communication operator.
    Expectation: success
    """
    return_code = os.system("mpirun --allow-run-as-root -n 2 pytest -s test_all_to_all_v_c.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_func_all_to_all_v_c():
    """
    Feature: mpi run 2P case of 'func_all_to_all_v_c' communication operator.
    Description: mpi run 2P case of 'func_all_to_all_v_c' communication operator.
    Expectation: success
    """
    return_code = os.system("mpirun --allow-run-as-root -n 2 pytest -s test_func_all_to_all_v_c.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_func_all_to_all_v_c1():
    """
    Feature: mpi run 2P case of 'func_all_to_all_v_c' communication operator.
    Description: mpi run 2P case of 'func_all_to_all_v_c' communication operator.
    Expectation: success
    """
    return_code = os.system("mpirun --allow-run-as-root -n 2 pytest -s test_func_all_to_all_v_c.py")
    return_code = os.system(
        r"cp  test_func_all_to_all_v_c.py test_func_all_to_all_v_c1.py && "\
        r"sed -i 's/mindspore\.communication\.comm_func\.distributed/mindspore.ops.communication/g' "\
        r"test_func_all_to_all_v_c1.py && msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 "\
        r"--master_port=10666 --join=True pytest -s test_func_all_to_all_v_c1.py"
    )
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_uneven_net():
    """
    Feature: mpi run 2P case of 'reduce_scatter_tensor_v' communication operator.
    Description: mpi run 2P case of 'reduce_scatter_tensor_v' communication operator.
    Expectation: success
    """
    return_code = os.system("mpirun --allow-run-as-root -n 2 pytest -s test_uneven_net.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_hccl_all_to_all_v():
    """
    Feature: mpi run 2P case of 'alltoallv' communication operator.
    Description: mpi run 2P case of 'alltoallv' communication operator.
    Expectation: success
    """
    return_code = os.system("mpirun --allow-run-as-root -n 2 pytest -s test_all_to_all_v.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_entry_hccl_allreduce_fusion_by_attr():
    """
    Feature: msrun allreduce fusion test case.
    Description: msrun allreduce fusion test case.
    Expectation: success
    """
    os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = str(1)
    os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = str(1)
    os.environ['HCCL_EXEC_TIMEOUT'] = str(300)
    os.environ['GLOG_v'] = str(1)
    return_code = os.system("rm -rf ～/ascend/log && rm -rf rank* && msrun --worker_num=8 --local_worker_num=8 "
                            "--join=True "
                            "pytest -s test_comm_fusion.py::test_hccl_allreduce_fusion_by_attr")
    if return_code != 0:
        import datetime
        t = datetime.datetime.now()
        f = t.strftime('%m-%d-%H-%M-%S')
        os.system(f"mkdir ~/zpc_{f} && cp -rf *.log ~/zpc_{f} && cp -rf ~/ascend/log ~/zpc_{f}")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_entry_hccl_allgather_fusion_by_attr():
    """
    Feature: msrun allgather fusion test case.
    Description: msrun allgather fusion test case.
    Expectation: success
    """
    return_code = os.system("rm -rf rank* && msrun --worker_num=8 --local_worker_num=8 --join=True "
                            "pytest -s test_comm_fusion.py::test_hccl_allgather_fusion_by_attr")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_entry_hccl_reducescatter_fusion_by_attr():
    """
    Feature: msrun reducescatter fusion test case.
    Description: msrun reducescatter fusion test case.
    Expectation: success
    """
    return_code = os.system("rm -rf rank* && msrun --worker_num=8 --local_worker_num=8 --join=True "
                            "pytest -s test_comm_fusion.py::test_hccl_reducescatter_fusion_by_attr")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_hccl_multi_stream():
    """
    Feature: multiple stream of hccl.
    Description: test assign stream based on communication domain.
    Expectation: expect correct result.
    """
    os.environ['MS_DEV_RUNTIME_CONF'] = 'multi_stream:group'
    return_code = os.system("mpirun --allow-run-as-root -n 8 "
                            "pytest -s ""test_multi_stream.py::test_hccl_multi_stream")
    assert return_code == 0
    del os.environ['MS_DEV_RUNTIME_CONF']


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_comm_multi_stream():
    """
    Feature: multiple stream of hccl.
    Description: test assign stream based on communication domain.
    Expectation: expect correct result.
    """
    return_code = os.system("msrun --worker_num=8 --local_worker_num=8 --join=True "
                            "pytest -s test_multi_stream.py::test_comm_multi_stream")
    assert return_code == 0

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_comm_multi_stream_raise_error():
    """
    Feature: multiple stream of hccl.
    Description: test assign stream based on communication domain.
    Expectation: expect correct result.
    """
    return_code = os.system("msrun --worker_num=8 --local_worker_num=8 --join=True "
                            "pytest -s test_multi_stream.py::test_comm_multi_stream_raise_error")
    assert return_code == 0

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards",
        essential_mark="essential")
@test_utils.run_test_with_On
def test_ops_all_gather_into_tensor_net():
    """
    Feature: mpi run 2P case of 'all_gather_into_tensor' dynamic and static unified ops.
    Description: mpi run 2P case of 'all_gather_into_tensor' dynamic and static unified ops.
    Expectation: success
    """
    return_code = os.system("msrun --worker_num=2 --local_worker_num=2 --join=True "
                            "pytest -s test_ops_all_gather_into_tensor.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards",
        essential_mark="essential")
@test_utils.run_test_with_On
def test_ops_all_gather_net():
    """
    Feature: mpi run 2P case of 'all_gather' dynamic and static unified ops.
    Description: mpi run 2P case of 'all_gather' dynamic and static unified ops.
    Expectation: success
    """
    return_code = os.system("msrun --worker_num=2 --local_worker_num=2 --join=True "
                            "pytest -s test_ops_all_gather.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_ops_all_reduce_net():
    """
    Feature: mpi run 2P case of 'test_ops_all_reduce' dynamic and static unified ops.
    Description: mpi run 2P case of 'test_ops_all_reduce' dynamic and static unified ops.
    Expectation: success
    """
    return_code = os.system("msrun --worker_num=2 --local_worker_num=2 --join=True "
                            "pytest -s test_ops_all_reduce.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_ops_isend_irecv_net():
    """
    Feature: mpi run 2P case of 'test_ops_isend_irecv' dynamic and static unified ops.
    Description: mpi run 2P case of 'test_ops_isend_irecv' dynamic and static unified ops.
    Expectation: success
    """
    return_code = os.system("msrun --worker_num=2 --local_worker_num=2 --join=True "
                            "pytest -s test_ops_isend_irecv.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_ops_reduce_scatter_tensor_net():
    """
    Feature: mpi run 2P case of 'test_ops_reduce_scatter_tensor' dynamic and static unified ops.
    Description: mpi run 2P case of 'test_ops_reduce_scatter_tensor' dynamic and static unified ops.
    Expectation: success
    """
    return_code = os.system("msrun --worker_num=2 --local_worker_num=2 --join=True "
                            "pytest -s test_ops_reduce_scatter_tensor.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_ops_reduce_scatter_net():
    """
    Feature: mpi run 2P case of 'test_ops_reduce_scatter' dynamic and static unified ops.
    Description: mpi run 2P case of 'test_ops_reduce_scatter' dynamic and static unified ops.
    Expectation: success
    """
    return_code = os.system("msrun --worker_num=2 --local_worker_num=2 --join=True "
                            "pytest -s test_ops_reduce_scatter.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_ops_all_to_all_single_net():
    """
    Feature: mpi run 2P case of 'test_ops_all_to_all_single' dynamic and static unified ops.
    Description: mpi run 2P case of 'test_ops_all_to_all_single' dynamic and static unified ops.
    Expectation: success
    """
    return_code = os.system("msrun --worker_num=2 --local_worker_num=2 --join=True "
                            "pytest -s test_ops_all_to_all_single.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards",
        essential_mark="essential")
@test_utils.run_test_with_On
def test_alltoall_grad_net():
    """
    Feature: mpi run 2P case of 'test_grad_alltoall' dynamic and static unified ops.
    Description: mpi run 2P case of 'test_grad_alltoall' dynamic and static unified ops.
    Expectation: success
    """
    return_code = os.system("msrun --worker_num=2 --local_worker_num=2 --join=True "
                            "pytest -s test_grad_alltoall.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_grad_allgather_net():
    """
    Feature: mpi run 2P case of 'test_grad_alltoall' dynamic and static unified ops.
    Description: mpi run 2P case of 'test_grad_alltoall' dynamic and static unified ops.
    Expectation: success
    """
    return_code = os.system("msrun --worker_num=2 --local_worker_num=2 --join=True "
                            "pytest -s test_grad_allgather.py")
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_grad_allreduce_net():
    """
    Feature: mpi run 2P case of 'test_grad_allreduce' dynamic and static unified ops.
    Description: mpi run 2P case of 'test_grad_allreduce' dynamic and static unified ops.
    Expectation: success
    """
    return_code = os.system("msrun --worker_num=2 --local_worker_num=2 --join=True "
                            "pytest -s test_grad_allreduce.py")
    assert return_code == 0
