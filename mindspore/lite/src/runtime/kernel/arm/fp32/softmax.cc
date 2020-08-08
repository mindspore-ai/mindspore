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

#include "src/runtime/kernel/arm/fp32/softmax.h"
#include <string.h>
#include <vector>
#include "src/runtime/kernel/arm/nnacl/fp32/softmax.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SoftMax;

namespace mindspore::kernel {
int SoftmaxCPUKernel::Init() {
  SoftmaxBaseCPUKernel::Init();

  // malloc tmp buffer
  auto axis = softmax_param_->axis_;
  sum_data = reinterpret_cast<float *>(malloc(softmax_param_->input_shape_[axis] * sizeof(float)));
  memset(sum_data, 0, softmax_param_->input_shape_[axis] * sizeof(float));
  return RET_OK;
}

int SoftmaxCPUKernel::ReSize() { return RET_OK; }

int SoftmaxCPUKernel::Run() {
  auto input_ptr = reinterpret_cast<float *>(inputs_.at(kInputIndex)->Data());
  auto output_ptr = reinterpret_cast<float *>(outputs_.at(kOutputIndex)->Data());
  Softmax(input_ptr, output_ptr, sum_data, softmax_param_);
  return RET_OK;
}

}  // namespace mindspore::kernel
