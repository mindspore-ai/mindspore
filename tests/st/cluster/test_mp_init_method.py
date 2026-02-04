# Copyright 2026 Huawei Technologies Co., Ltd
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
This test is for multiprocess launch using init_method.
"""
import os
import numpy as np
import mindspore.multiprocessing as mp
import mindspore as ms
from mindspore import context
from mindspore.ops import cat
from mindspore.mint.distributed import (
    all_gather_into_tensor,
    init_process_group,
)
from mindspore.communication import (
    create_group,
    get_rank,
    get_local_rank,
    get_group_size,
    get_local_rank_size,
)


def run_init_method(rank, env_dict=None):
    if env_dict is not None:
        os.environ.update(env_dict)

    context.set_context(mode=context.PYNATIVE_MODE)
    context.set_context(jit_level="O0")
    context.set_context(device_target="Ascend")

    print(f"===== Begin task! pid: {os.getpid()}  rank: {rank} =====", flush=True)

    if os.getenv("INIT_DEFAULT_GROUP"):
        if rank in [0, 1, 2, 3, 4, 5]:
            ms.set_device(device_target="Ascend", device_id=rank)
            group_rank = rank
            worker_num = 6
            init_process_group(
                init_method="tcp://127.0.0.1:12333",
                rank=group_rank,
                world_size=worker_num,
            )

            # group rank: [0, 1, 2, 3, 4, 5]
            assert get_rank() == rank
            assert get_local_rank() == rank
            assert get_group_size() == 6
            assert get_local_rank_size() == 6

        if rank in [6, 7]:
            ms.set_device(device_target="Ascend", device_id=rank)
            group_rank = rank % 6
            worker_num = 2
            init_process_group(
                init_method="tcp://127.0.0.1:12444",
                rank=group_rank,
                world_size=worker_num,
            )

            # group rank: [0, 1]
            assert get_rank() == rank % 6
            assert get_local_rank() == rank % 6
            assert get_group_size() == 2
            assert get_local_rank_size() == 2

    # Initialize the cross-default communication group when 'INIT_CROSS_DEFAULT_GROUP' is configured.
    if os.getenv("INIT_CROSS_DEFAULT_GROUP"):
        if rank in [0, 6]:
            ms.set_device(device_target="Ascend", device_id=rank)
            group_rank = rank % 5
            worker_num = 2
            init_group_name = "init_for_06"
            init_process_group(
                init_method="tcp://127.0.0.1:12555",
                rank=group_rank,
                world_size=worker_num,
                group_name=init_group_name,
            )

            # group rank: [0, 1]
            assert get_rank(init_group_name) == rank % 5
            assert get_local_rank(init_group_name) == rank % 5
            assert get_group_size(init_group_name) == worker_num
            assert get_local_rank_size(init_group_name) == worker_num

            if not os.getenv("INIT_CROSS_DEFAULT_GROUP"):
                # Only initialize group with group_name rather than initializing default group;
                # get_rank(init_group_name) == get_rank().
                assert get_rank() == rank % 5
                assert get_local_rank() == rank % 5
                assert get_group_size() == worker_num
                assert get_local_rank_size() == worker_num

        if rank in [4, 7]:
            ms.set_device(device_target="Ascend", device_id=rank)
            group_rank = (rank - 4) // 3
            worker_num = 2
            init_group_name = "init_for_47"
            init_process_group(
                init_method="tcp://127.0.0.1:12666",
                rank=group_rank,
                world_size=worker_num,
                group_name=init_group_name,
            )

            # group rank: [0, 1]
            assert get_rank(init_group_name) == (rank - 4) // 3
            assert get_local_rank(init_group_name) == (rank - 4) // 3
            assert get_group_size(init_group_name) == worker_num
            assert get_local_rank_size(init_group_name) == worker_num

            if not os.getenv("INIT_CROSS_DEFAULT_GROUP"):
                # Only initialize group with group_name rather than initializing default group;
                # get_rank(init_group_name) == get_rank().
                assert get_rank() == (rank - 4) // 3
                assert get_local_rank() == (rank - 4) // 3
                assert get_group_size() == worker_num
                assert get_local_rank_size() == worker_num

    if os.getenv("INIT_DEFAULT_GROUP"):
        # 1.Only initialize the default group; 2.Initialize the default group first, then the cross group.
        if rank in [0, 4]:
            assert get_rank() == rank
            assert get_local_rank() == rank
            assert get_group_size() == 6
            assert get_local_rank_size() == 6

        if rank in [6, 7]:
            assert get_rank() == rank % 6
            assert get_local_rank() == rank % 6
            assert get_group_size() == 2
            assert get_local_rank_size() == 2

    if os.getenv("INIT_DEFAULT_GROUP"):
        create_group_name = "self_group"
        if rank in [0, 1, 4, 5]:
            create_group(create_group_name, [0, 1, 4, 5])
            # group rank: [0, 1, 2, 3]
            assert get_rank(create_group_name) == rank - 2 * (rank // 4)
            assert get_local_rank(create_group_name) == rank - 2 * (rank // 4)
            assert get_group_size(create_group_name) == 4
            assert get_local_rank_size(create_group_name) == 4

            input_tensor = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
            output_tensor = ms.Tensor(np.zeros([12, 3]).astype(np.float32))
            output_handle = all_gather_into_tensor(
                output_tensor, input_tensor, group=create_group_name
            )
            except_output_tensor = cat(
                [input_tensor, input_tensor, input_tensor, input_tensor]
            )
            assert output_handle is None
            assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())

        if rank in [0, 1, 2, 3, 4, 5]:
            input_tensor = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
            output_tensor = ms.Tensor(np.zeros([18, 3]).astype(np.float32))
            output_handle = all_gather_into_tensor(output_tensor, input_tensor)
            except_output_tensor = cat(
                [
                    input_tensor,
                    input_tensor,
                    input_tensor,
                    input_tensor,
                    input_tensor,
                    input_tensor,
                ]
            )
            assert output_handle is None
            assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())

        if rank in [6, 7]:
            input_tensor = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
            output_tensor = ms.Tensor(np.zeros([6, 3]).astype(np.float32))
            output_handle = all_gather_into_tensor(output_tensor, input_tensor)
            except_output_tensor = cat([input_tensor, input_tensor])
            assert output_handle is None
            assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())

    if os.getenv("INIT_CROSS_DEFAULT_GROUP"):
        if rank in [0, 6]:
            input_tensor = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
            output_tensor = ms.Tensor(np.zeros([6, 3]).astype(np.float32))
            output_handle = all_gather_into_tensor(output_tensor, input_tensor, group="init_for_06")
            except_output_tensor = cat([input_tensor, input_tensor])
            assert output_handle is None
            assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())

        if rank in [4, 7]:
            input_tensor = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
            output_tensor = ms.Tensor(np.zeros([6, 3]).astype(np.float32))
            output_handle = all_gather_into_tensor(output_tensor, input_tensor, group="init_for_47")
            except_output_tensor = cat([input_tensor, input_tensor])
            assert output_handle is None
            assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())

    print(f"===== End task! pid: {os.getpid()}  rank: {rank} =====", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    rank_size = 8
    process_list = []

    for i in range(rank_size):
        env = os.environ.copy()
        new_env = {**env}

        p = mp.Process(target=run_init_method, args=(i, new_env))

        process_list.append(p)
        p.start()
        print(f"Launch rank{i}, pid: {p.pid}.", flush=True)

    for p in process_list:
        p.join()

    for idx, p in enumerate(process_list):
        if p.exitcode != 0:
            raise RuntimeError(
                f"rank: {idx}  pid: {p.pid} exits with exception! Error code: {p.exitcode}."
            )
