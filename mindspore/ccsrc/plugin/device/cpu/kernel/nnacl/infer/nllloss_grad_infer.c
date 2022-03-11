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

#include "nnacl/infer/nllloss_grad_infer.h"

#include "nnacl/infer/infer_register.h"

int NLLLossGradInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                          OpParameter *parameter) {
  int ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, C5NUM, C1NUM);
  if (ret != NNACL_OK) {
    return ret;
  }

  const TensorC *logits = inputs[0];
  const TensorC *loss_grad = inputs[1];
  const TensorC *labels = inputs[2];
  const TensorC *weight = inputs[3];
  const TensorC *total_weight = inputs[4];
  if (logits->shape_size_ != C2NUM || labels->shape_size_ != C1NUM || weight->shape_size_ != C1NUM ||
      total_weight->shape_size_ != 0) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  if (labels->shape_[0] != logits->shape_[0] || weight->shape_[0] != logits->shape_[1]) {
    return NNACL_INPUT_TENSOR_ERROR;
  }

  NLLLossParameter *param = (NLLLossParameter *)parameter;
  if (param->reduction_type_ == Reduction_None && loss_grad->shape_size_ != C1NUM) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  if (param->reduction_type_ != Reduction_None && loss_grad->shape_size_ != 0) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  TensorC *logits_grad = outputs[0];
  SetDataTypeFormat(logits_grad, logits);
  SetShapeTensor(logits_grad, logits);
  return NNACL_OK;
}

REG_INFER(NLLLossGrad, PrimType_NLLLossGrad, NLLLossGradInferShape)
