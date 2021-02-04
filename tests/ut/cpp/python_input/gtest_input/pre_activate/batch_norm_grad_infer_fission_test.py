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
from mindspore.ops import Primitive
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops import _constants as Constants

make_tuple = Primitive('MakeTuple')
tuple_getitem = Primitive(Constants.kTupleGetItem)
BatchNormGradTraining = G.BatchNormGrad(is_training=True)
BatchNormGradInfer = G.BatchNormGrad(is_training=False)
BNInferGrad = Primitive('BNInferGrad')
BNTrainingUpdateGrad = Primitive('BNTrainingUpdateGrad')


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_batch_norm_grad_infer_fission(tag):
    fns = FnDict()

    @fns
    def before(input0, input1, input2, input3, input4, input5):
        batch_norm = BatchNormGradInfer(input0, input1, input2, input3, input4, input5)
        outputs = make_tuple(tuple_getitem(batch_norm, 0), tuple_getitem(batch_norm, 1), tuple_getitem(batch_norm, 2))
        output = tuple_getitem(outputs, 0)
        return output

    @fns
    def before_is_training(input0, input1, input2, input3, input4, input5):
        batch_norm = BatchNormGradTraining(input0, input1, input2, input3, input4, input5)
        outputs = make_tuple(tuple_getitem(batch_norm, 0), tuple_getitem(batch_norm, 1), tuple_getitem(batch_norm, 2))
        output = tuple_getitem(outputs, 0)
        return output

    @fns
    def before_output3_not_null(input0, input1, input2, input3, input4, input5):
        batch_norm = BatchNormGradInfer(input0, input1, input2, input3, input4, input5)
        outputs = make_tuple(tuple_getitem(batch_norm, 0), tuple_getitem(batch_norm, 1), tuple_getitem(batch_norm, 2))
        output = tuple_getitem(outputs, 0)
        return output

    @fns
    def after(input0, input1, input2, input3, input4, input5):
        bn_infer_grad = BNInferGrad(input0, input2, input4)
        bn_training_update_grad = BNTrainingUpdateGrad(input0, input1, input3, input4)
        outputs = make_tuple(bn_infer_grad, tuple_getitem(bn_training_update_grad, 0),
                             tuple_getitem(bn_training_update_grad, 1))
        new_outputs = make_tuple(tuple_getitem(outputs, 0), tuple_getitem(outputs, 1), tuple_getitem(outputs, 2))
        output = tuple_getitem(new_outputs, 0)
        return make_tuple(output)

    return fns[tag]
