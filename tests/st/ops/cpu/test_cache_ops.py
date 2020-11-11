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
import math
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import Parameter
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE,
                    device_target='CPU', save_graphs=True)


def hash_func(key, length):
    return (int)(((0.6180339 * key) - math.floor(0.6180339 * key)) * length)


def init_hashmap(hash_map_length):
    key_np = np.array([2, 3, 10, 15, 21], np.int32)
    value_np = np.array([1, 3, 5, 7, 9], np.int32)
    NULLTAG = 0
    INIT_STEP = -5
    hashmap_np = np.zeros((hash_map_length, 4), np.int32)
    for i, key in enumerate(key_np):
        entry = hash_func(key, hash_map_length)
        count = 1
        while (hashmap_np[entry, 3] != NULLTAG and hashmap_np[entry, 0] != key):
            count += 1
            entry = (entry + 1) % hash_map_length
        if (hashmap_np[entry, 3] == NULLTAG):
            hashmap_np[entry] = [key, value_np[i], INIT_STEP, count]

    return hashmap_np


class SearchCacheIdxNet(nn.Cell):
    def __init__(self, hashmap_np):
        super().__init__()
        self.ops = P.SearchCacheIdx()
        self.hashmap = Parameter(Tensor(hashmap_np), name="hashmap")
        self.emb_max = 25
        self.cache_max = 10
        self.step = 0

    def construct(self, indices):
        return self.ops(self.hashmap, indices, self.step, self.emb_max, self.cache_max)


class CacheSwapHashmapNet(nn.Cell):
    def __init__(self, hashmap_np):
        super().__init__()
        self.net = SearchCacheIdxNet(hashmap_np)
        self.ops = P.CacheSwapHashmap()
        self.step = 0
        self.emb_max = 25
        self.cache_max = 10

    def construct(self, indices):
        _, _, miss_emb_idx = self.net(indices)
        return self.ops(self.net.hashmap, miss_emb_idx, self.step)


class UpdateCacheNet(nn.Cell):
    def __init__(self, x):
        super().__init__()
        self.ops = P.UpdateCache()
        self.max_num = 9999
        self.x = Parameter(Tensor(x), name='x')

    def construct(self, indices, update):
        return self.ops(self.x, indices, update, self.max_num)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_search_cache_idx():
    hashmap_np = init_hashmap(10)
    indices_np = np.array([10, 2, 20, 5, 3], np.int32)
    search_cache_idx = SearchCacheIdxNet(hashmap_np)
    indices = Tensor(indices_np)
    cache_idx, miss_idx, miss_emb_idx = search_cache_idx(indices)

    expect_cache_idx = [5, 1, -1, -1, 3]
    expect_miss_idx = [-1, -1, 2, 3, -1]
    expect_miss_emb_idx = [-1, -1, 20, 5, -1]

    hashmap_np_after_ops = [[0, 0, 0, 0],
                            [10, 5, 0, 1],
                            [2, 1, 0, 1],
                            [15, 7, -5, 2],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [3, 3, 0, 1],
                            [21, 9, -5, 1]]

    assert np.allclose(cache_idx.asnumpy(),
                       np.array(expect_cache_idx, np.int32))
    assert np.allclose(miss_idx.asnumpy(), np.array(expect_miss_idx, np.int32))
    assert np.allclose(miss_emb_idx.asnumpy(),
                       np.array(expect_miss_emb_idx, np.int32))
    assert np.allclose(search_cache_idx.hashmap.data.asnumpy(),
                       np.array(hashmap_np_after_ops, np.int32))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cache_swap_hashmap():
    hashmap_np = init_hashmap(10)
    indices_np = np.array([10, 2, 20, 5, 3], np.int32)
    net = CacheSwapHashmapNet(hashmap_np)
    indices = Tensor(indices_np)
    swap_cache_idx, old_emb_idx = net(indices)

    expect_swap_cache_idx = [-1, -1, 9, 7, -1]
    expect_old_emb_idx = [-1, -1, 21, 15, -1]

    hashmap_np_after_ops = [[5, 7, 0, 1],
                            [10, 5, 0, 1],
                            [2, 1, 0, 1],
                            [20, 9, 0, 1],
                            [20, 9, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [3, 3, 0, 1],
                            [21, 9, -5, 0]]

    assert np.allclose(swap_cache_idx.asnumpy(),
                       np.array(expect_swap_cache_idx, np.int32))
    assert np.allclose(old_emb_idx.asnumpy(),
                       np.array(expect_old_emb_idx, np.int32))
    assert np.allclose(net.net.hashmap.data.asnumpy(),
                       np.array(hashmap_np_after_ops, np.int32))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_update_cache():
    x_np = np.array([[2, 3, 4, 5],
                     [6, 7, 8, 9],
                     [11, 12, 13, 14],
                     [1, 2, 3, 4],
                     [5, 6, 7, 8]], np.int32)

    indices_np = np.array([[-1, 3, 4]], np.int32)
    update_np = np.array([[0, 0, 0, 0],
                          [23, 34, 56, 78],
                          [44, 55, 66, 77]], np.int32)

    indices = Tensor(indices_np)
    update = Tensor(update_np)

    expect = np.array([[2, 3, 4, 5],
                       [6, 7, 8, 9],
                       [11, 12, 13, 14],
                       [23, 34, 56, 78],
                       [44, 55, 66, 77]], np.int32)
    net = UpdateCacheNet(x_np)
    out = net(indices, update)
    assert np.allclose(net.x.data.asnumpy(), expect)
    assert np.allclose(out.asnumpy(), np.array([0], np.int32))
