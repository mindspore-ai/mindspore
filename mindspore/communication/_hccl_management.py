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
"""HCCL management API"""
import ctypes

MAX_GROUP_NAME_LEN = 127
MAX_RANK_NUM = 4096
HCCL_LIB = 'libhccl.so'
HCCL_LIB_CTYPES = ""


def check_group(group):
    """
    A function that check if a collection communication group is leagal.

    Returns:
        None
    """
    if isinstance(group, (str)):
        group_len = len(group)
        if group_len > MAX_GROUP_NAME_LEN or group_len == 0:
            raise ValueError('Group name is invalid.')
    else:
        raise TypeError('Group must be a python str.')


def check_rank_num(rank_num):
    """
    A function that check if a collection communication rank number is leagal.If not raise error.

    Returns:
        None
    """
    if isinstance(rank_num, (int)):
        if rank_num > MAX_RANK_NUM or rank_num <= 0:
            raise ValueError('Rank number is out of range.')
    else:
        raise TypeError('Rank number must be a python int.')


def check_rank_id(rank_id):
    """
    A function that check if a collection communication rank id is leagal.If not raise error.

    Returns:
        None
    """
    if isinstance(rank_id, (int)):
        if rank_id >= MAX_RANK_NUM or rank_id < 0:
            raise ValueError('Rank id is out of range.')
    else:
        raise TypeError('Rank id must be a python int.')


def load_lib():
    try:
        hccl_lib = ctypes.CDLL(HCCL_LIB)
    except Exception:
        raise RuntimeError('Get hccl lib error.')
    global HCCL_LIB_CTYPES
    HCCL_LIB_CTYPES = hccl_lib


def c_str(string):
    """Convert a python string to C string."""
    if not isinstance(string, str):
        string = string.decode('ascii')
    return ctypes.c_char_p(string.encode('utf-8'))


def c_array(ctype, values):
    """Create ctypes array from a python array."""
    return (ctype * len(values))(*values)


def create_group(group, rank_num, rank_ids):
    """
    Create group.

    A function that creates a collection communication group which includes 'rank_num'
    device and 'rank_ids' is the list of these ranks of devices.

    Note:
        The world group can not be created.

    Returns:
        None
    """
    check_group(group)
    check_rank_num(rank_num)
    if isinstance(rank_ids, (list)):
        if rank_num != len(rank_ids):
            raise ValueError('Rank number is not equal to the length of rank_ids.')
        for rank_id in rank_ids:
            if not isinstance(rank_id, (int)) or rank_id < 0:
                raise ValueError('Rank id must be unsigned integer!')
        c_array_rank_ids = c_array(ctypes.c_uint, rank_ids)
        c_rank_num = ctypes.c_uint(rank_num)
        c_group = c_str(group)
        ret = HCCL_LIB_CTYPES.HcomCreateGroup(c_group, c_rank_num, c_array_rank_ids)
        if ret != 0:
            raise RuntimeError('Create group error.')
    else:
        raise TypeError('Rank ids must be a python list.')


def destroy_group(group):
    """
    A function that destroy the group which created by user.

    Note:
        The world group can not be destroy.

    Returns:
        None
    """
    check_group(group)
    c_group = c_str(group)
    ret = HCCL_LIB_CTYPES.HcomDestroyGroup(c_group)
    if ret != 0:
        raise RuntimeError('Destroy group error.')


def get_rank_size(group="hccl_world_group"):
    """
    A function that returns the number of ranks within the given collection communication group.

    Note:
        The default group is hccl_world_group.

    Returns:
        An integer scalar with the num of ranks.
    """
    check_group(group)
    c_group = c_str(group)
    c_rank_size = ctypes.c_uint()
    ret = HCCL_LIB_CTYPES.HcomGetRankSize(c_group, ctypes.byref(c_rank_size))
    if ret != 0:
        raise RuntimeError('Get rank size error.')

    return c_rank_size.value


def get_rank_id(group="hccl_world_group"):
    """
    A function that returns the rank id of the calling process, within the given collection communication group.

    Returns:
        An integer scalar with the rank id of the calling process.
    """
    check_group(group)
    c_group = c_str(group)
    c_rank_id = ctypes.c_uint()
    ret = HCCL_LIB_CTYPES.HcomGetRankId(c_group, ctypes.byref(c_rank_id))
    if ret != 0:
        raise RuntimeError('Get rank id error.')

    return c_rank_id.value


def get_local_rank_size(group="hccl_world_group"):
    """
    A function that returns the number of local ranks within the given collection communication group.

    Note:
        The default group is hccl_world_group.

    Returns:
        An integer scalar with the num of local ranks.
    """
    check_group(group)
    c_group = c_str(group)
    c_local_rank_size = ctypes.c_uint()
    ret = HCCL_LIB_CTYPES.HcomGetLocalRankSize(c_group, ctypes.byref(c_local_rank_size))
    if ret != 0:
        raise RuntimeError('Get local rank size error.')

    return c_local_rank_size.value


def get_local_rank_id(group="hccl_world_group"):
    """
    Get local rank id.

    A function that returns the local rank id of the calling process, within the given collection communication group.

    Returns:
        An integer scalar with the local rank id of the calling process.
    """
    check_group(group)
    c_group = c_str(group)
    c_local_rank_id = ctypes.c_uint()
    ret = HCCL_LIB_CTYPES.HcomGetLocalRankId(c_group, ctypes.byref(c_local_rank_id))
    if ret != 0:
        raise RuntimeError('Get local rank id error.')

    return c_local_rank_id.value


def get_world_rank_from_group_rank(group, group_rank_id):
    """
    Get world rank from group rank.

    A function that returns the rank id in the world group corresponding to the
    rank which id is 'group_rank_id' in the user group.

    Returns:
        An integer scalar with the rank id in the world group.
    """
    check_group(group)
    check_rank_id(group_rank_id)
    c_group = c_str(group)
    c_group_rank_id = ctypes.c_uint(group_rank_id)
    c_world_rank_id = ctypes.c_uint()
    ret = HCCL_LIB_CTYPES.HcomGetWorldRankFromGroupRank(c_group, c_group_rank_id, ctypes.byref(c_world_rank_id))
    if ret != 0:
        raise RuntimeError('Get world rank from group rank error.')

    return c_world_rank_id.value


def get_group_rank_from_world_rank(world_rank_id, group):
    """
    Get group rank from world rank.

    A function that returns the rank id in the user group corresponding to the
    rank which id is 'world_rank_id' in the world group.

    Returns:
        An integer scalar with the rank id in the user group.
    """
    check_group(group)
    check_rank_id(world_rank_id)
    c_group = c_str(group)
    c_world_rank_id = ctypes.c_uint(world_rank_id)
    c_group_rank_id = ctypes.c_uint()
    ret = HCCL_LIB_CTYPES.HcomGetGroupRankFromWorldRank(c_world_rank_id, c_group, ctypes.byref(c_group_rank_id))
    if ret != 0:
        raise RuntimeError('Get group rank from world rank error.')

    return c_group_rank_id.value
