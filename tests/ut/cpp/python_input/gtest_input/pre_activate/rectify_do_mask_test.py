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
# ============================================================================
from mindspore.ops import operations as P

shape = (2, 4, 2, 2)
gen_mask = P.DropoutGenMask(10, 10)
do_mask = P.DropoutDoMask()


class FnDict:
    def __init__(self):
        self.fn_dict = {}

    def __call__(self, fn):
        self.fn_dict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fn_dict.get(name)


def test_rectify_dropout_do_mask(tag):
    """
    Feature: Test RectifyDoMaskKernelInfo pass
    Description: Test RectifyDoMaskKernelInfo pass
    Expectation: The forward and backward DropOutDoMask should select same format.
    """
    fns = FnDict()

    @fns
    def f(x, y):
        mask = gen_mask(shape, 0.9)
        res_x = do_mask(x, mask, 0.9)
        res_y = do_mask(y, mask, 0.9)
        return res_x, res_y

    return fns[tag]
