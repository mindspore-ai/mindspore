/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "internal/src/kernel/fp32_grad/activation_grad.h"
#include "internal/src/kernel/common/common_infershape.h"
#include "internal/include/errorcode.h"
#include "internal/include/ms_tensor.h"
#include "nnacl/fp32_grad/activation_grad.h"
#include "internal/src/lite_log.h"
#include "nnacl/errorcode.h"

int DoActivationGradInferShape(const TensorPtrVector &in_tensors, const TensorPtrVector &out_tensors,
                               OpParameter *param) {
  return DoCommonInferShape(in_tensors, out_tensors);
}

int DoActivationGrad(const TensorPtrVector &in_tensors, const TensorPtrVector &out_tensors, Node *node,
                     mindspore::lite::Allocator *allocator) {
  ActivationGradParameter *param = (ActivationGradParameter *)node->primitive_;
  int ret = RET_OK;
  size_t length = in_tensors[0]->ElementsNum();
  float *dy_data = (float *)in_tensors[0]->data_;
  float *x_data = (float *)in_tensors[1]->data_;
  float *dx_data = (float *)(float *)out_tensors[0]->data_;
  if (param->type_ == ActivationType::RELU) {
    ret = ReluGrad(dy_data, x_data, length, dx_data);
  } else if (param->type_ == ActivationType::SIGMOID) {
    ret = SigmoidGrad(dy_data, x_data, length, dx_data);
  } else if (param->type_ == ActivationType::RELU6) {
    ret = Relu6Grad(dy_data, x_data, length, dx_data);
  } else if (param->type_ == ActivationType::LEAKY_RELU) {
    float alpha = param->alpha_;
    ret = LReluGrad(dy_data, x_data, length, dx_data, alpha);
  } else {
    LITE_ERROR_LOG("Unsupport activation type %d", param->type_);
    return RET_PARAM_INVALID;
  }
  if (ret != NNACL_OK) {
    LITE_ERROR_LOG("do activation(%d) fail!ret: %d", param->type_, ret);
    return RET_ERROR;
  }
  return RET_OK;
}
