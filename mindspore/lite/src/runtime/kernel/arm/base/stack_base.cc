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
#include "src/runtime/kernel/arm/base/stack_base.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "nnacl/base/stack_base.h"
#include "nnacl/stack_parameter.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Stack;

namespace mindspore::kernel {
static inline int GetCopyNum(const std::vector<int> &in_shape, int axis, int n_dim) {
  int copy_num = 1;
  if (axis > 0) {
    for (int j = n_dim - 1; j > axis - 1; j--) {
      copy_num *= in_shape[j];
    }
  } else {
    for (int i = 0; i < n_dim; ++i) {
      copy_num *= in_shape[i];
    }
  }
  return copy_num;
}

static inline size_t GetOuterSize(const std::vector<int> &in_shape, int axis) {
  size_t outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= in_shape[i];
  }
  return outer_size;
}

int StackBaseCPUKernel::ReSize() {
  auto param = reinterpret_cast<StackParameter *>(op_parameter_);
  auto input0_shape = in_tensors_.front()->shape();
  axis_ = param->axis_ < 0 ? param->axis_ + input0_shape.size() + 1 : param->axis_;
  auto input_nums = in_tensors_.size();
  if (input_nums == 1) {
    copy_size_ = in_tensors_.front()->ElementsNum() * data_type_size_;
  } else {
    MS_ASSERT(input_nums > 1);
    copy_size_ = GetCopyNum(input0_shape, axis_, input0_shape.size()) * data_type_size_;
    outer_size_ = GetOuterSize(input0_shape, axis_);
  }
  return RET_OK;
}

int StackBaseCPUKernel::Init() {
  data_type_size_ = sizeof(float);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int StackBaseCPUKernel::Run() {
  // malloc temporary memory to store all the inputs
  size_t inputs_num = in_tensors_.size();
  char **all_inputs = static_cast<char **>(context_->allocator->Malloc(inputs_num * sizeof(char *)));
  if (all_inputs == nullptr) {
    MS_LOG(ERROR) << "malloc all_inputs failed.";
    return RET_ERROR;
  }
  for (size_t j = 0; j < inputs_num; ++j) {
    all_inputs[j] = reinterpret_cast<char *>(in_tensors_.at(j)->data_c());
  }
  // run stack
  auto output_data = reinterpret_cast<char *>(out_tensors_.at(0)->data_c());
  Stack(all_inputs, output_data, in_tensors_.size(), copy_size_, outer_size_);

  // free temporary variable all_inputs
  context_->allocator->Free(all_inputs);
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Stack, LiteKernelCreator<StackBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Stack, LiteKernelCreator<StackBaseCPUKernel>)
}  // namespace mindspore::kernel
