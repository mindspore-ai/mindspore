# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore.ops import operations as P

add = P.Add()
mul = P.Mul()
all_reduce = P.AllReduce().add_prim_attr("fusion", 1)


def add_net(x1, x2, x3, x4, x5):
    sum1 = add(x1, x2)
    sum2 = add(sum1, x3)
    sum3 = add(sum2, x4)
    sum4 = add(sum3, x5)
    ret = mul(sum4, sum1)
    return ret


def all_reduce_net(x1, x2, x3):
    product = mul(x1, x2)
    sum1 = add(x2, x3)
    reduce1 = all_reduce(product)
    reduce2 = all_reduce(sum1)
    res = add(reduce1, reduce2)
    return res


def add_with_all_reduce_net(x1, x2, x3, x4, x5):
    a1 = all_reduce(add(x1, x2))
    a2 = all_reduce(add(x2, x3))
    a3 = all_reduce(add(x3, x4))
    sum1 = add(a1, x3)
    sum2 = add(a2, x4)
    sum3 = add(a3, x5)
    sum4 = add(sum1, sum2)
    ret = mul(sum3, sum4)
    return ret
