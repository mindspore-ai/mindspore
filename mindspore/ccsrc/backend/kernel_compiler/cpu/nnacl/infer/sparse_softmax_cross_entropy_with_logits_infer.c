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

#include "nnacl/infer/sparse_softmax_cross_entropy_with_logits_infer.h"
#include "nnacl/fp32_grad/softmax_grad.h"
#include "nnacl/infer/infer_register.h"

int SparseSoftmaxCrossEntropyWithLogitsInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs,
                                                  size_t outputs_size, OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 2, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *in0 = inputs[0];
  TensorC *out = outputs[0];

  SoftmaxCrossEntropyParameter *param = (SoftmaxCrossEntropyParameter *)parameter;
  if (param->is_grad_ != 0) {
    SetShapeTensor(out, in0);
    SetDataTypeFormat(out, in0);
  } else {
    out->shape_size_ = 1;
    out->shape_[0] = 1;
    SetDataTypeFormat(out, in0);
  }

  return NNACL_OK;
}

REG_INFER(SparseSoftmaxCrossEntropyWithLogits, PrimType_SparseSoftmaxCrossEntropyWithLogits,
          SparseSoftmaxCrossEntropyWithLogitsInferShape)
