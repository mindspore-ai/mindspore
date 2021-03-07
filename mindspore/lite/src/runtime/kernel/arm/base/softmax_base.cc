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

#include "src/runtime/kernel/arm/base/softmax_base.h"
#include <vector>
#include "src/runtime/kernel/arm/fp32/softmax_fp32.h"
#include "nnacl/fp32/softmax_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {

int SoftmaxBaseCPUKernel::Init() {
  if (softmax_param_ == nullptr) {
    MS_LOG(ERROR) << "SoftmaxParameter nullptr";
    return RET_NULL_PTR;
  }
  return RET_OK;
}

int SoftmaxBaseCPUKernel::ReSize() {
  auto input_tensor = in_tensors_.front();
  auto in_shape = input_tensor->shape();
  auto in_dims = in_shape.size();
  int ele_size = 1;
  softmax_param_->n_dim_ = in_dims;
  if (softmax_param_->axis_ == -1) {
    softmax_param_->axis_ += in_dims;
  }
  for (size_t i = 0; i < in_dims; i++) {
    softmax_param_->input_shape_[i] = in_shape.at(i);
    ele_size *= in_shape.at(i);
  }
  softmax_param_->element_size_ = ele_size;
  return RET_OK;
}
}  // namespace mindspore::kernel
