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

import numpy as np
from mindspore.parallel._tensor import _transform_tensor_by_layout, _get_needed_rank_list_by_layouts, \
    _get_needed_rank_transform_operator_map_by_layouts, _generate_transform_operator_stack, \
    _apply_tensor_transform_operators, _construct_from_to_tensor_layout, _construct_tensor_layout_for_opt_shard, \
    _get_tensor_strategy


def test_transform_tensor_by_layout_allconcat_axis_1():
    """
    Feature: transform tensor by layout.
    Description: allconcat.
    Expectation: assert no error.
    """
    from_layout = ((1, 8), (1, 0), (32, 256))
    to_layout = ((1, 4, 2), (2, 1), (32, 256))
    device_list = [0, 1, 2, 3, 4, 5, 6, 7]
    rank_id = 0
    op_list = _transform_tensor_by_layout(from_layout, to_layout, device_list, rank_id)
    assert op_list == [('Reshape', [32, 1, 32]), ('AllGather', [0, 1, 2]), ('Reshape', [32, 64])]


def test_transform_tensor_by_layout_allconcat_axis_1_using_none_map():
    """
    Feature: transform tensor by layout.
    Description: allconcat, tensor map contains -1
    Expectation: assert no error.
    """
    from_layout = ((8,), (0, -1), (32, 256))
    to_layout = ((4, 2), (1, -1), (32, 256))
    device_list = [0, 1, 2, 3, 4, 5, 6, 7]
    rank_id = 0
    op_list = _transform_tensor_by_layout(from_layout, to_layout, device_list, rank_id)
    assert op_list == [('Reshape', [1, 4, 256]), ('AllGather', [0, 1, 1]), ('Reshape', [8, 256])]


def test_transform_tensor_by_layout_allconcat_to_single():
    """
    Feature: transform tensor by layout.
    Description: allconcat, transform to single device.
    Expectation: assert no error.
    """
    from_layout = ((4, 2), (1, 0), (32, 256))
    to_layout = ((1, 8), (-1, -1), (32, 256))
    device_list = [0, 1, 2, 3, 4, 5, 6, 7]
    rank_id = 0
    op_list = _transform_tensor_by_layout(from_layout, to_layout, device_list, rank_id)
    assert op_list == [('AllGather', [0, 2, 4, 6, 0]), ('AllGather', [0, 1, 1])]


def test_transform_tensor_by_layout_allconcat_axis_0():
    """
    Feature: transform tensor by layout.
    Description: allconcat.
    Expectation: assert no error.
    """
    from_layout = ((8, 1), (1, 0), (32, 256))
    to_layout = ((2, 1, 4), (2, 1), (32, 256))
    device_list = [0, 1, 2, 3, 4, 5, 6, 7]
    rank_id = 0
    op_list = _transform_tensor_by_layout(from_layout, to_layout, device_list, rank_id)
    assert op_list == [('Reshape', [1, 4, 256]), ('AllGather', [0, 1, 2, 3, 1]), ('Reshape', [16, 256])]


def test_transform_tensor_by_layout_all_to_all():
    """
    Feature: transform tensor by layout.
    Description: all to all.
    Expectation: assert no error.
    """
    from_layout = ((8, 1), (1, -1), (32, 64))
    to_layout = ((1, 8), (-1, 0), (32, 64))
    device_list = list(range(0, 8))
    rank_id = 0
    op_list = _transform_tensor_by_layout(from_layout, to_layout, device_list, rank_id)
    assert op_list == [('AllGather', [0, 1, 2, 3, 4, 5, 6, 7, 0]), ('StridedSlice', [0, 0, 32, 8, 1, 1])]


def test_transform_tensor_by_layout_mix():
    """
    Feature: transform tensor by layout.
    Description: allconcat + allsplit.
    Expectation: assert no error.
    """
    from_layout = ((2, 2, 2), (2, 1, 0), (32, 64, 128))
    to_layout = ((8, 1, 1), (2, 1, 0), (32, 64, 128))
    device_list = [0, 1, 2, 3, 4, 5, 6, 7]
    rank_id = 1
    op_list = _transform_tensor_by_layout(from_layout, to_layout, device_list, rank_id)
    assert op_list == [('Reshape', [1, 2, 8, 32, 64]), ('AllGather', [1, 3, 3]),
                       ('StridedSlice', [0, 0, 0, 0, 0, 1, 1, 8, 64, 64, 1, 1, 1, 1, 1]),
                       ('AllGather', [0, 1, 4]), ('StridedSlice', [0, 0, 4, 0, 0, 1, 1, 8, 64, 128, 1, 1, 1, 1, 1]),
                       ('Reshape', [4, 64, 128])]


def test_needed_rank_list_by_layouts_1():
    """
    Feature: get needed rank list for transform tensor by layout.
    Description: allconcat + allsplit.
    Expectation: assert no error.
    """
    from_layout = ((2, 2, 2), (2, 1, 0), (32, 64, 128))
    to_layout = ((8, 1, 1), (2, 1, 0), (32, 64, 128))
    device_list = [0, 1, 2, 3, 4, 5, 6, 7]
    rank_id = 1
    needed_rank_list = _get_needed_rank_list_by_layouts(from_layout, to_layout, device_list, rank_id)
    assert needed_rank_list == [0, 1, 2, 3]


def test_needed_rank_list_by_layouts_2():
    """
    Feature: get needed rank list for transform tensor by layout.
    Description: allconcat + allsplit, 128p.
    Expectation: assert no error.
    """
    from_layout = ((32, 1, 8), (2, -1, 0), (32, 64, 128))
    to_layout = ((2, 64, 2), (2, 1, 0), (32, 64, 128))
    device_list = list(range(0, 256))
    rank_id = 1
    needed_rank_list = _get_needed_rank_list_by_layouts(from_layout, to_layout, device_list, rank_id)
    assert needed_rank_list == list(range(0, 128))


def test_generate_transform_operator_stack_1():
    """
    Feature: generate transform operator stack.
    Description: moe transform.
    Expectation: assert no error.
    """
    from_layout = ((8, 1, 8), (2, 1, -1), (32, 64, 128))
    to_layout = ((8, 8, 1), (2, 1, 0), (32, 64, 128))
    device_list = list(range(0, 64))
    param_rank_map = _get_needed_rank_transform_operator_map_by_layouts(from_layout, to_layout,
                                                                        device_list, 0)
    assert param_rank_map == {0: [('StridedSlice', [0, 0, 0, 4, 8, 128, 1, 1, 1])]}
    transform_operator_stack = _generate_transform_operator_stack(param_rank_map, 0)
    assert transform_operator_stack == [(0, 0, ('StridedSlice', [0, 0, 0, 4, 8, 128, 1, 1, 1]))]


def test_generate_transform_operator_stack_2():
    """
    Feature: generate transform operator stack.
    Description: all to all.
    Expectation: assert no error.
    """
    from_layout = ((8, 1), (1, -1), (32, 64))
    to_layout = ((1, 8), (-1, 0), (32, 64))
    device_list = list(range(0, 8))
    param_rank_map = _get_needed_rank_transform_operator_map_by_layouts(from_layout, to_layout,
                                                                        device_list, 0)
    assert len(param_rank_map) == 8
    transform_operator_stack = _generate_transform_operator_stack(param_rank_map, 0)
    assert transform_operator_stack == [(0, 1, ('StridedSlice', [0, 0, 32, 8, 1, 1])),
                                        (0, 0, ('AllGather', [0, 1, 2, 3, 4, 5, 6, 7, 0]))]


def test_generate_transform_operator_stack_3():
    """
    Feature: generate transform operator stack.
    Description: mix.
    Expectation: assert no error.
    """
    from_layout = ((8, 1, 8), (2, -1, 0), (32, 64, 128))
    to_layout = ((2, 8, 4), (2, 1, 0), (32, 64, 128))
    device_list = list(range(0, 64))
    param_rank_map = _get_needed_rank_transform_operator_map_by_layouts(from_layout, to_layout,
                                                                        device_list, 0)
    transform_operator_stack = _generate_transform_operator_stack(param_rank_map, 0)
    assert len(transform_operator_stack) == 79


def test_generate_transform_operator_stack_4():
    """
    Feature: generate transform operator stack.
    Description: multi allconcat and allsplit.
    Expectation: assert no error.
    """
    from_layout = ((2, 2, 2), (2, 1, 0), (32, 64, 128))
    to_layout = ((8, 1, 1), (2, 1, 0), (32, 64, 128))
    device_list = [0, 1, 2, 3, 4, 5, 6, 7]
    rank_id = 1
    param_rank_map = _get_needed_rank_transform_operator_map_by_layouts(from_layout, to_layout,
                                                                        device_list, rank_id)
    transform_operator_stack = _generate_transform_operator_stack(param_rank_map, rank_id)
    assert transform_operator_stack == [(1, 5, ('Reshape', [4, 64, 128])),
                                        (1, 4, ('StridedSlice', [0, 0, 4, 0, 0, 1, 1, 8, 64, 128, 1, 1, 1, 1, 1])),
                                        (1, 3, ('AllGather', [0, 1, 4])),
                                        (0, 2, ('StridedSlice', [0, 0, 0, 0, 0, 1, 1, 8, 64, 64, 1, 1, 1, 1, 1])),
                                        (1, 2, ('StridedSlice', [0, 0, 0, 0, 0, 1, 1, 8, 64, 64, 1, 1, 1, 1, 1])),
                                        (0, 1, ('AllGather', [0, 2, 3])), (1, 1, ('AllGather', [1, 3, 3])),
                                        (0, 0, ('Reshape', [1, 2, 8, 32, 64])), (2, 0, ('Reshape', [1, 2, 8, 32, 64])),
                                        (1, 0, ('Reshape', [1, 2, 8, 32, 64])), (3, 0, ('Reshape', [1, 2, 8, 32, 64]))]


def test_apply_tensor_transform_operators_allconcat():
    """
    Feature: apply tensor transform operators.
    Description: allconcat.
    Expectation: assert no error.
    """
    device_num = 8
    tensor_dict = {}
    for rank in range(device_num):
        tensor_dict[rank] = np.full((1, 8, 8), rank)
    from_layout = ((8, 1, 1), (2, 1, 0), (8, 8, 8))
    to_layout = ((1, 1, 1, 8), (-1, -1, -1), (8, 8, 8))
    device_list = [0, 1, 2, 3, 4, 5, 6, 7]
    rank_id = 0
    param_rank_map = _get_needed_rank_transform_operator_map_by_layouts(from_layout, to_layout,
                                                                        device_list, rank_id)
    transform_operator_stack = _generate_transform_operator_stack(param_rank_map, rank_id)
    _apply_tensor_transform_operators(transform_operator_stack, tensor_dict, device_num)
    assert tensor_dict.get(0).shape == (8, 8, 8)
    for rank in range(8):
        assert np.all(tensor_dict.get(0)[rank] == rank)


def test_apply_tensor_transform_operators_allsplit():
    """
    Feature: apply tensor transform operators.
    Description: allsplit.
    Expectation: assert no error.
    """
    device_num = 8
    tensor_dict = {}
    for rank in range(device_num):
        tensor_dict[rank] = np.array([np.full((8, 8), i) for i in range(device_num)])
    from_layout = ((8,), (-1, -1, -1), (8, 8, 8))
    to_layout = ((8,), (-1, -1, 0), (8, 8, 8))
    device_list = [0, 1, 2, 3, 4, 5, 6, 7]
    rank_id = 0
    param_rank_map = _get_needed_rank_transform_operator_map_by_layouts(from_layout, to_layout,
                                                                        device_list, rank_id)
    transform_operator_stack = _generate_transform_operator_stack(param_rank_map, rank_id)
    _apply_tensor_transform_operators(transform_operator_stack, tensor_dict, device_num)
    assert tensor_dict.get(0).shape == (8, 8, 1)
    for rank in range(8):
        assert np.all(tensor_dict.get(0)[rank] == rank)


def test_apply_tensor_transform_operators_mix():
    """
    Feature: apply tensor transform operators.
    Description: mulit allconcat, allsplit.
    Expectation: assert no error.
    """
    device_num = 8
    tensor_dict = {}
    for rank in range(device_num):
        tensor_dict[rank] = np.full((1, 8, 8), rank)
    from_layout = ((8, 1, 1), (2, 1, 0), (8, 8, 8))
    to_layout = ((2, 2, 2), (2, 1, 0), (8, 8, 8))
    device_list = [0, 1, 2, 3, 4, 5, 6, 7]
    rank_id = 0
    param_rank_map = _get_needed_rank_transform_operator_map_by_layouts(from_layout, to_layout,
                                                                        device_list, rank_id)
    transform_operator_stack = _generate_transform_operator_stack(param_rank_map, rank_id)
    _apply_tensor_transform_operators(transform_operator_stack, tensor_dict, device_num)
    assert tensor_dict.get(0).shape == (4, 4, 4)
    for rank in range(4):
        assert np.all(tensor_dict.get(0)[rank] == rank)


def test_apply_tensor_transform_operators_no_need_transform():
    """
    Feature: apply tensor transform operators.
    Description: no need transform.
    Expectation: assert no error.
    """
    device_num = 8
    tensor_dict = {}
    for rank in range(device_num):
        tensor_dict[rank] = np.full((1, 8, 8), rank)
    from_layout = ((8, 1, 1), (2, -1, -1), (8, 8, 8))
    to_layout = ((8, 1, 1), (2, -1, -1), (8, 8, 8))
    device_list = [0, 1, 2, 3, 4, 5, 6, 7]
    rank_id = 0
    param_rank_map = _get_needed_rank_transform_operator_map_by_layouts(from_layout, to_layout,
                                                                        device_list, rank_id)
    transform_operator_stack = _generate_transform_operator_stack(param_rank_map, rank_id)
    _apply_tensor_transform_operators(transform_operator_stack, tensor_dict, device_num)
    assert tensor_dict.get(0).shape == (1, 8, 8)
    assert np.all(tensor_dict.get(0) == rank_id)


def test_construct_tensor_layout_for_opt_shard():
    """
    Feature: construct tensor layout for optimizer shard.
    Description: construct tensor layout for optimizer shard.
    Expectation: assert no error.
    """
    dev_matrix = (2, 2, 2)
    tensor_map = (2, 1, -1)
    opt_shard_step = 4
    opt_shard_size = 2
    origin_full_tensor_shape = (16, 16, 16)
    new_dev_matrix, new_tensor_map, new_shape = _construct_tensor_layout_for_opt_shard(dev_matrix, tensor_map,
                                                                                       opt_shard_step, opt_shard_size,
                                                                                       origin_full_tensor_shape)
    assert new_dev_matrix == [2, 2, 2, 1]
    assert new_tensor_map == [2, 3, 1, -1]
    assert new_shape == [2, 8, 16, 16]


def test_construct_from_to_tensor_layout():
    """
    Feature: construct from and to tensor layout.
    Description: construct from and to tensor layout.
    Expectation: assert no error.
    """
    tensor_shape = (8, 1024)
    from_dev_matrix = (8, 8)
    from_tensor_map = (-1, 0)
    to_dev_matrix = (16, 8)
    to_tensor_map = (1, -1)
    from_tensor_layout, to_tensor_layout = _construct_from_to_tensor_layout(tensor_shape, from_dev_matrix,
                                                                            from_tensor_map, tensor_shape,
                                                                            to_dev_matrix, to_tensor_map)
    assert from_tensor_layout == ([2, 8, 8], [-1, 0], [8, 1024])
    assert to_tensor_layout == ([16, 8], [1, -1], [8, 1024])


def conver_tensor_by_layout(from_dev_matrix, from_tensor_map, from_opt_shard_step, from_opt_shard_size,
                            to_dev_matrix_origin, to_tensor_map_origin, to_opt_shard_step, to_opt_shard_size,
                            tensor_dict, rank_id):

    device_num = np.prod(from_dev_matrix)
    tensor_shape = tensor_dict[rank_id % device_num].shape
    param_strategy = _get_tensor_strategy(from_dev_matrix, from_tensor_map)
    origin_tensor_shape = ()
    for i, item in enumerate(tensor_shape):
        if i == 0 and from_opt_shard_size > 0:
            origin_tensor_shape += (item * param_strategy[i] * from_opt_shard_size,)
            continue
        origin_tensor_shape += (item * param_strategy[i],)

    from_dev_matrix, from_tensor_map, from_full_tensor_shape = _construct_tensor_layout_for_opt_shard(
        from_dev_matrix, from_tensor_map, from_opt_shard_step, from_opt_shard_size, origin_tensor_shape)
    to_dev_matrix, to_tensor_map, to_full_tensor_shape = _construct_tensor_layout_for_opt_shard(
        to_dev_matrix_origin, to_tensor_map_origin, to_opt_shard_step, to_opt_shard_size, origin_tensor_shape)
    # Convert tensor layout to same device num
    from_tensor_layout, to_tensor_layout = _construct_from_to_tensor_layout(from_full_tensor_shape, from_dev_matrix,
                                                                            from_tensor_map, to_full_tensor_shape,
                                                                            to_dev_matrix, to_tensor_map)

    # when the from_layout is less devices, the checkpoint_map for map[device_num] should using map[0]

    device_list = list(range(0, np.prod(from_tensor_layout[0])))
    if rank_id % device_num not in tensor_dict:
        raise ValueError("The checkpoint of rank {} is missing.".format(rank_id % device_num))
    param_rank_map = _get_needed_rank_transform_operator_map_by_layouts(from_tensor_layout, to_tensor_layout,
                                                                        device_list, rank_id)
    for param_rank, _ in param_rank_map.items():
        if from_opt_shard_size > 0:
            from_tensor_strategy = _get_tensor_strategy(from_dev_matrix, from_tensor_map)
            from_slice_tensor_shape = ()
            for i, item in enumerate(from_full_tensor_shape):
                from_slice_tensor_shape += (item // from_tensor_strategy[i],)
            param_rank_map.get(param_rank).insert(0, ('Reshape', list(from_slice_tensor_shape)))
        if to_opt_shard_size > 0:
            to_tensor_strategy = _get_tensor_strategy(to_dev_matrix_origin, to_tensor_map_origin)
            to_slice_tensor_shape = ()
            for i, item in enumerate(origin_tensor_shape):
                if i == 0 and to_opt_shard_size > 0:
                    to_slice_tensor_shape += (item // (to_tensor_strategy[i] * to_opt_shard_size),)
                    continue
                to_slice_tensor_shape += (item // to_tensor_strategy[i],)
            param_rank_map.get(param_rank).append(('Reshape', list(to_slice_tensor_shape)))

    transform_operator_stack = _generate_transform_operator_stack(param_rank_map, rank_id)
    _apply_tensor_transform_operators(transform_operator_stack, tensor_dict, device_num)

    return tensor_dict[rank_id % device_num]


def test_transform_parallel_checkpoint():
    """
    Feature: transform parallel checkpoint.
    Description: device_num 16. None -> optimizer_shard 2, model_parallel 4
        -> optimizer_shard 4, model_parallel 2 -> optimizer_shard 16, model_parallel 1
    Expectation: assert no error.
    """
    import copy
    device_num = 16
    tensor_dict = {}
    for rank in range(device_num):
        tensor_dict[rank] = np.array([np.full((8,), i) for i in range(device_num)])
    no_change_tensor_dict = copy.deepcopy(tensor_dict)
    result_dict = {}
    from_dev_matrix = (16,)
    from_tensor_map = (-1, -1)
    from_opt_shard_step = 0
    from_opt_shard_size = 0
    to_dev_matrix = (4, 4)
    to_tensor_map = (0, -1)
    to_opt_shard_step = 4
    to_opt_shard_size = 2
    for rank_id in range(device_num):
        result = conver_tensor_by_layout(from_dev_matrix, from_tensor_map, from_opt_shard_step, from_opt_shard_size,
                                         to_dev_matrix, to_tensor_map, to_opt_shard_step, to_opt_shard_size,
                                         tensor_dict, rank_id)
        result_dict[rank_id] = result
        tensor_dict = copy.deepcopy(no_change_tensor_dict)
        rank = rank_id % 8
        first_value = (rank % 4) * 4 + (rank // 4) * 2
        assert np.all(result[0] == first_value)
        assert np.all(result[1] == first_value + 1)
    to_dev_matrix1 = (8, 2)
    to_tensor_map1 = (0, -1)
    to_opt_shard_step1 = 2
    to_opt_shard_size1 = 4
    tensor_dict = copy.deepcopy(result_dict)
    no_change_tensor_dict = copy.deepcopy(result_dict)
    for rank_id in range(device_num):
        result = conver_tensor_by_layout(to_dev_matrix, to_tensor_map, to_opt_shard_step, to_opt_shard_size,
                                         to_dev_matrix1, to_tensor_map1, to_opt_shard_step1, to_opt_shard_size1,
                                         tensor_dict, rank_id)
        result_dict[rank_id] = result
        tensor_dict = copy.deepcopy(no_change_tensor_dict)
        rank = rank_id % 8
        first_value = (rank % 2) * 8 + (rank // 2) * 2
        assert np.all(result[0] == first_value)
        assert np.all(result[1] == first_value + 1)
    to_dev_matrix2 = (16,)
    to_tensor_map2 = (-1, -1)
    to_opt_shard_step2 = 1
    to_opt_shard_size2 = 16
    tensor_dict = copy.deepcopy(result_dict)
    no_change_tensor_dict = copy.deepcopy(result_dict)
    for rank_id in range(device_num):
        result = conver_tensor_by_layout(to_dev_matrix1, to_tensor_map1, to_opt_shard_step1, to_opt_shard_size1,
                                         to_dev_matrix2, to_tensor_map2, to_opt_shard_step2, to_opt_shard_size2,
                                         tensor_dict, rank_id)
        result_dict[rank_id] = result
        tensor_dict = copy.deepcopy(no_change_tensor_dict)
        assert np.all(result == rank_id)


def test_transform_parallel_checkpoint_1():
    """
    Feature: transform parallel checkpoint.
    Description: model_parallel in last dim. device_num 16. None -> optimizer_shard 2, model_parallel 4
        -> optimizer_shard 4, model_parallel 2 -> optimizer_shard 16, model_parallel 1
    Expectation: assert no error.
    """
    import copy
    device_num = 16
    tensor_dict = {}
    for rank in range(device_num):
        tensor_dict[rank] = np.array([np.full((8,), i) for i in range(device_num)])
    no_change_tensor_dict = copy.deepcopy(tensor_dict)
    result_dict = {}
    from_dev_matrix = (16,)
    from_tensor_map = (-1, -1)
    from_opt_shard_step = 0
    from_opt_shard_size = 0
    to_dev_matrix = (4, 4)
    to_tensor_map = (-1, 0)
    to_opt_shard_step = 4
    to_opt_shard_size = 2
    for rank_id in range(device_num):
        result = conver_tensor_by_layout(from_dev_matrix, from_tensor_map, from_opt_shard_step, from_opt_shard_size,
                                         to_dev_matrix, to_tensor_map, to_opt_shard_step, to_opt_shard_size,
                                         tensor_dict, rank_id)
        result_dict[rank_id] = result
        tensor_dict = copy.deepcopy(no_change_tensor_dict)
        rank = rank_id % 8
        first_value = (rank // 4) * 8
        assert np.all(result[0] == first_value)
    to_dev_matrix1 = (8, 2)
    to_tensor_map1 = (-1, 0)
    to_opt_shard_step1 = 2
    to_opt_shard_size1 = 4
    tensor_dict = copy.deepcopy(result_dict)
    no_change_tensor_dict = copy.deepcopy(result_dict)
    for rank_id in range(device_num):
        result = conver_tensor_by_layout(to_dev_matrix, to_tensor_map, to_opt_shard_step, to_opt_shard_size,
                                         to_dev_matrix1, to_tensor_map1, to_opt_shard_step1, to_opt_shard_size1,
                                         tensor_dict, rank_id)
        result_dict[rank_id] = result
        tensor_dict = copy.deepcopy(no_change_tensor_dict)
        rank = rank_id % 8
        first_value = (rank // 2) * 4
        assert np.all(result[0] == first_value)
    to_dev_matrix2 = (16,)
    to_tensor_map2 = (-1, -1)
    to_opt_shard_step2 = 1
    to_opt_shard_size2 = 16
    tensor_dict = copy.deepcopy(result_dict)
    no_change_tensor_dict = copy.deepcopy(result_dict)
    for rank_id in range(device_num):
        result = conver_tensor_by_layout(to_dev_matrix1, to_tensor_map1, to_opt_shard_step1, to_opt_shard_size1,
                                         to_dev_matrix2, to_tensor_map2, to_opt_shard_step2, to_opt_shard_size2,
                                         tensor_dict, rank_id)
        result_dict[rank_id] = result
        tensor_dict = copy.deepcopy(no_change_tensor_dict)
        assert np.all(result == rank_id)


def test_transform_parallel_checkpoint_2():
    """
    Feature: transform parallel checkpoint.
    Description: model_parallel in last dim. device_num 16. None -> device_num 8, optimizer_shard 2, model_parallel 4
        -> device_num 16, optimizer_shard 4, model_parallel 2.
    Expectation: assert no error.
    """
    import copy
    device_num = 16
    tensor_dict = {}
    for rank in range(device_num):
        tensor_dict[rank] = np.array([np.full((8,), i) for i in range(device_num)])
    no_change_tensor_dict = copy.deepcopy(tensor_dict)
    result_dict = {}
    from_dev_matrix = (16,)
    from_tensor_map = (-1, -1)
    from_opt_shard_step = 0
    from_opt_shard_size = 0
    to_dev_matrix = (2, 4)
    to_tensor_map = (0, -1)
    to_opt_shard_step = 4
    to_opt_shard_size = 2
    for rank_id in range(8):
        result = conver_tensor_by_layout(from_dev_matrix, from_tensor_map, from_opt_shard_step, from_opt_shard_size,
                                         to_dev_matrix, to_tensor_map, to_opt_shard_step, to_opt_shard_size,
                                         tensor_dict, rank_id)
        result_dict[rank_id] = result
        tensor_dict = copy.deepcopy(no_change_tensor_dict)
        rank = rank_id % 8
        first_value = (rank % 4) * 4 + (rank // 4) * 2
        assert np.all(result[0] == first_value)
        assert np.all(result[1] == first_value + 1)
    to_dev_matrix1 = (8, 2)
    to_tensor_map1 = (0, -1)
    to_opt_shard_step1 = 2
    to_opt_shard_size1 = 4
    tensor_dict = copy.deepcopy(result_dict)
    no_change_tensor_dict = copy.deepcopy(result_dict)
    for rank_id in range(device_num):
        result = conver_tensor_by_layout(to_dev_matrix, to_tensor_map, to_opt_shard_step, to_opt_shard_size,
                                         to_dev_matrix1, to_tensor_map1, to_opt_shard_step1, to_opt_shard_size1,
                                         tensor_dict, rank_id)
        result_dict[rank_id] = result
        tensor_dict = copy.deepcopy(no_change_tensor_dict)
        rank = rank_id % 8
        first_value = (rank % 2) * 8 + (rank // 2) * 2
        assert np.all(result[0] == first_value)
        assert np.all(result[1] == first_value + 1)
