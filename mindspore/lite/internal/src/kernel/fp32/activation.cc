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

#include "internal/src/kernel/fp32/activation.h"
#include "internal/src/kernel/common/common_infershape.h"
#include "internal/include/errorcode.h"
#include "internal/include/ms_tensor.h"
#include "nnacl/fp32/activation.h"
#include "internal/src/lite_log.h"
#include "nnacl/errorcode.h"

int DoActivationInferShape(const TensorPtrVector &in_tensors, const TensorPtrVector &out_tensors, OpParameter *param) {
  return DoCommonInferShape(in_tensors, out_tensors);
}

int DoActivation(const TensorPtrVector &in_tensors, const TensorPtrVector &out_tensors, Node *node,
                 mindspore::lite::Allocator *allocator) {
  ActivationParameter *param = (ActivationParameter *)node->primitive_;
  int ret = RET_OK;
  size_t length = in_tensors[0]->ElementsNum();
  float *input_addr = (float *)in_tensors[0]->data_;
  float *output_addr = (float *)out_tensors[0]->data_;
  if (param->type_ == ActivationType::RELU) {
    ret = Fp32Relu(input_addr, length, output_addr);
  } else if (param->type_ == ActivationType::SIGMOID) {
    ret = Sigmoid(input_addr, length, output_addr);
  } else if (param->type_ == ActivationType::RELU6) {
    ret = Fp32Relu6(input_addr, length, output_addr);
  } else if (param->type_ == ActivationType::LEAKY_RELU) {
    float alpha = param->alpha_;
    ret = LRelu(input_addr, length, output_addr, alpha);
  } else {
    LITE_ERROR_LOG("Unsupport activation type: %d", param->type_);
    return RET_PARAM_INVALID;
  }
  if (ret != NNACL_OK) {
    LITE_ERROR_LOG("do activation(%d) fail!ret: %d", param->type_, ret);
    return RET_ERROR;
  }
  return RET_OK;
}
