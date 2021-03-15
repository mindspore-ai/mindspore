/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nnacl/infer/softmax_cross_entropy_infer.h"
#include "nnacl/infer/infer_register.h"

int SoftmaxCrossEntropyInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs,
                                  size_t outputs_size, OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  if (1 > outputs_size) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
#endif

  const TensorC *in0 = inputs[0];
  TensorC *out = outputs[0];

  out->shape_size_ = 2;
  out->shape_[0] = in0->shape_[0];
  out->shape_[1] = 1;
  SetDataTypeFormat(out, in0);

  if (1 < outputs_size) {
    TensorC *grads = outputs[1];
    SetShapeTensor(grads, in0);
    SetDataTypeFormat(grads, in0);
  }
  return NNACL_OK;
}

REG_INFER(SoftmaxCrossEntropyWithLogits, PrimType_SoftmaxCrossEntropyWithLogits, SoftmaxCrossEntropyInferShape)
