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

"""
management api
"""
import mindspore.communication.management as D

def has_raise_error(func, x):
    try:
        # pylint:disable=eval-used
        if x is None:
            func()
        else:
            func(x)
        print("x:{}".format(x))
    except (TypeError, ValueError, RuntimeError):
        return True
    else:
        return False

def create_backend(name):
    D.Backend(name)

def get_group_size_int(group):
    D.get_group_size(group)

def create_group0(x):
    D.GlobalComm.BACKEND = D.Backend.HCCL
    D.create_group('0-1', x)

def create_group1(x):
    D.GlobalComm.BACKEND = D.Backend.HCCL
    D.create_group('0-1', x)

def create_group2(x):
    D.GlobalComm.BACKEND = D.Backend.HCCL
    D.create_group('0-1', x)

def create_group3(x):
    D.GlobalComm.BACKEND = D.Backend.UNDEFINED
    D.create_group('0-1', x)

def create_group4(x):
    D.GlobalComm.BACKEND = D.Backend.HCCL
    D.create_group('0-1', x)

def get_world_rank_from_group_rank0():
    D.GlobalComm.BACKEND = D.Backend.HCCL
    D.get_world_rank_from_group_rank(D.HCCL_WORLD_COMM_GROUP, 0)

def get_world_rank_from_group_rank1():
    D.GlobalComm.BACKEND = D.Backend.HCCL
    D.get_world_rank_from_group_rank('0-1', '0')

def get_world_rank_from_group_rank2():
    D.GlobalComm.BACKEND = D.Backend.UNDEFINED
    D.get_world_rank_from_group_rank('0-1', 0)

def get_group_rank_from_world_rank0():
    D.GlobalComm.BACKEND = D.Backend.HCCL
    D.get_group_rank_from_world_rank(0, D.HCCL_WORLD_COMM_GROUP)

def get_group_rank_from_world_rank1():
    D.GlobalComm.BACKEND = D.Backend.HCCL
    D.get_group_rank_from_world_rank('0', '0-1')

def get_group_rank_from_world_rank2():
    D.GlobalComm.BACKEND = D.Backend.UNDEFINED
    D.get_group_rank_from_world_rank(0, '0-1')

def destroy_group0(x):
    D.GlobalComm.BACKEND = D.Backend.UNDEFINED
    D.destroy_group(x)

def destroy_group1():
    D.GlobalComm.BACKEND = D.Backend.HCCL
    D.destroy_group(D.HCCL_WORLD_COMM_GROUP)

def destroy_group2(x):
    D.GlobalComm.BACKEND = D.Backend.HCCL
    D.destroy_group(x)

def test_raise_error_funcs():
    """test raise error funcs"""
    assert has_raise_error(create_backend, 123) is True
    assert has_raise_error(create_backend, 'hccl') is False
    assert has_raise_error(create_backend, 'nccl') is False
    assert has_raise_error(get_group_size_int, 123) is True
    assert has_raise_error(create_group0, (0,1)) is True
    assert has_raise_error(create_group1, [0]) is False
    assert has_raise_error(create_group2, [0,0,1]) is True
    assert has_raise_error(create_group3, [0,1]) is True
    assert has_raise_error(create_group4, [0,1]) is False
    assert has_raise_error(get_world_rank_from_group_rank0, None) is True
    assert has_raise_error(get_world_rank_from_group_rank1, None) is True
    assert has_raise_error(get_world_rank_from_group_rank2, None) is True
    assert has_raise_error(get_group_rank_from_world_rank0, None) is True
    assert has_raise_error(get_group_rank_from_world_rank1, None) is True
    assert has_raise_error(get_group_rank_from_world_rank2, None) is True
    assert has_raise_error(destroy_group0, '0-1') is True
    assert has_raise_error(destroy_group1, None) is True
    assert has_raise_error(destroy_group2, '0-1') is False

def test_get_rank_none():
    assert D.get_rank(group=None) == 0

def test_group_funs():
    D.GlobalComm.BACKEND = D.Backend.HCCL
    assert D.get_group_size(group=None) == 1
    assert D.get_group_size('2-abcd') == 2
    assert D.get_world_rank_from_group_rank('0-1', 0) == 0
    assert D.get_group_rank_from_world_rank(0, '0-1') == 0

