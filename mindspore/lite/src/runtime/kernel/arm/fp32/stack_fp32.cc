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
#include "src/runtime/kernel/arm/fp32/stack_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "nnacl/fp32/stack_fp32.h"
#include "nnacl/stack_parameter.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Stack;

namespace mindspore::kernel {
int StackCPUKernel::ReSize() {
  StackParameter *param = reinterpret_cast<StackParameter *>(op_parameter_);
  auto input0_shape = in_tensors_.at(0)->shape();
  axis_ = param->axis_ < 0 ? param->axis_ + input0_shape.size() + 1 : param->axis_;
  return RET_OK;
}

int StackCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int StackCPUKernel::Run() {
  size_t inputs_num = in_tensors_.size();
  auto input0 = in_tensors_.at(0);
  if (inputs_num == 1) {
    auto *output_data = reinterpret_cast<int8_t *>(out_tensors_.at(0)->MutableData());
    MS_ASSERT(output_data);
    auto *input_data = reinterpret_cast<const int8_t *>(input0->MutableData());
    MS_ASSERT(input_data);
    DoStackOneInput(input_data, output_data, input0->Size());
    return RET_OK;
  }
  auto input0_shape = in_tensors_.at(0)->shape();
  if (in_tensors_.at(0)->data_type() == kNumberTypeFloat32 || in_tensors_.at(0)->data_type() == kNumberTypeFloat) {
    auto *output_data = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
    MS_ASSERT(output_data);
    float *inputs[inputs_num];
    for (size_t i = 0; i < inputs_num; ++i) {
      inputs[i] = reinterpret_cast<float *>(in_tensors_.at(i)->MutableData());
      MS_ASSERT(inputs[i]);
    }
    DoStack(inputs, inputs_num, input0_shape.data(), input0_shape.size(), axis_, output_data);
  } else {
    auto *output_data = reinterpret_cast<int32_t *>(out_tensors_.at(0)->MutableData());
    MS_ASSERT(output_data);
    int32_t *inputs[inputs_num];
    for (size_t i = 0; i < inputs_num; ++i) {
      inputs[i] = reinterpret_cast<int32_t *>(in_tensors_.at(i)->MutableData());
      MS_ASSERT(inputs[i]);
    }
    DoStackInt32(inputs, inputs_num, input0_shape.data(), input0_shape.size(), axis_, output_data);
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Stack, LiteKernelCreator<StackCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Stack, LiteKernelCreator<StackCPUKernel>)
}  // namespace mindspore::kernel
