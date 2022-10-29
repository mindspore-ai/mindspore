/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "nnacl/infer/split_reduce_concat_infer.h"
#include "nnacl/infer/infer_register.h"
#include "nnacl/split_parameter.h"

int SplitReduceConcatFusionInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs,
                                      size_t outputs_size, OpParameter *parameter) {
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  MS_CHECK_TRUE_RET(inputs_size == outputs_size, NNACL_INPUT_TENSOR_ERROR);
  const TensorC *in_tensor = inputs[0];
  TensorC *out_tensor = outputs[0];
  out_tensor->format_ = in_tensor->format_;
  for (size_t i = 0; i < in_tensor->shape_size_; i++) {
    out_tensor->shape_[i] = in_tensor->shape_[i];
  }
  SplitParameter *param = (SplitParameter *)parameter;
  out_tensor->shape_[param->split_dim_] = param->num_split_;
  out_tensor->shape_size_ = in_tensor->shape_size_;
  return NNACL_OK;
}

REG_INFER(SplitReduceConcatFusion, PrimType_Inner_SplitReduceConcatFusion, SplitReduceConcatFusionInferShape)
