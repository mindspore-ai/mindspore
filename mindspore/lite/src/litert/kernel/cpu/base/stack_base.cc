/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "src/litert/kernel/cpu/base/stack_base.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/base/stack_base.h"
#include "include/errorcode.h"
#include "nnacl/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
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

static inline int GetOuterSize(const std::vector<int> &in_shape, int axis) {
  int outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= in_shape[i];
  }
  return outer_size;
}

int StackBaseCPUKernel::ReSize() {
  CHECK_NULL_RETURN(in_tensors_.front());
  auto input0_shape = in_tensors_.front()->shape();
  axis_ =
    stack_param_->axis_ < 0 ? stack_param_->axis_ + static_cast<int>(input0_shape.size()) + 1 : stack_param_->axis_;
  auto input_nums = in_tensors_.size();
  if (input_nums == 1) {
    MS_CHECK_GT(in_tensors_.front()->ElementsNum(), 0, RET_ERROR);
    copy_size_ = in_tensors_.front()->ElementsNum() * data_type_size_;
  } else {
    CHECK_LESS_RETURN(input_nums, THIRD_INPUT);
    CHECK_LESS_RETURN(input0_shape.size(), static_cast<size_t>(axis_));
    copy_size_ = static_cast<size_t>(GetCopyNum(input0_shape, axis_, input0_shape.size())) * data_type_size_;
    outer_size_ = GetOuterSize(input0_shape, axis_);
  }

  if (UpdateThreadNumPass(TC_PTYPE(type_), copy_size_, copy_size_, out_tensors_.at(0)->ElementsNum()) != RET_OK) {
    return RET_ERROR;
  }
  thread_num_ = MSMIN(UP_DIV(outer_size_, 64), op_parameter_->thread_num_);  // 64 : stack step

  return RET_OK;
}

int StackBaseCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(stack_param_);
  auto data_type = in_tensors_.at(FIRST_INPUT)->data_type();
  if (data_type == kNumberTypeFloat32 || data_type == kNumberTypeInt32) {
    data_type_size_ = sizeof(float);
  } else {
    MS_LOG(ERROR) << "stack not support data type: " << data_type;
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int StackBaseCPUKernel::StackExecute(int task_id) {
  auto output_data = reinterpret_cast<void *>(out_tensors_.at(0)->data());
  if (output_data == nullptr) {
    return RET_NULL_PTR;
  }
  MS_CHECK_TRUE_RET(thread_num_ != 0, RET_ERROR);
  auto step = UP_DIV(outer_size_, thread_num_);
  MS_CHECK_FALSE(INT_MUL_OVERFLOW(task_id, step), RET_ERROR);
  auto start = task_id * step;
  auto end = MSMIN(start + step, outer_size_);
  auto input_num = in_tensors_.size();
  MS_CHECK_FALSE(INT_MUL_OVERFLOW(input_num * static_cast<size_t>(start), copy_size_), RET_ERROR);
  auto output = reinterpret_cast<char *>(output_data) + input_num * static_cast<size_t>(start) * copy_size_;
  Stack(all_inputs_, reinterpret_cast<void *>(output), input_num, copy_size_, start, end);
  return RET_OK;
}

static int StackRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto stack = reinterpret_cast<StackBaseCPUKernel *>(cdata);
  CHECK_NULL_RETURN(stack);
  if (stack->StackExecute(task_id) != RET_OK) {
    return RET_ERROR;
  }
  return RET_OK;
}

int StackBaseCPUKernel::Run() {
  // malloc temporary memory to store all the inputs
  size_t inputs_num = in_tensors_.size();
  all_inputs_ = static_cast<void **>(ms_context_->allocator->Malloc(inputs_num * sizeof(void *)));
  if (all_inputs_ == nullptr) {
    MS_LOG(ERROR) << "malloc all_inputs failed.";
    return RET_ERROR;
  }
  for (size_t j = 0; j < inputs_num; ++j) {
    auto input_data = reinterpret_cast<void *>(in_tensors_.at(j)->data());
    if (input_data == nullptr) {
      return RET_NULL_PTR;
    }
    all_inputs_[j] = input_data;
  }
  // run stack
  CHECK_NULL_RETURN(out_tensors_.at(0));

  auto ret = ParallelLaunch(this->ms_context_, StackRun, this, thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "StackBaseCPUKernel Run error: error_code[" << ret << "]";
    return RET_ERROR;
  }

  // free temporary variable all_inputs
  ms_context_->allocator->Free(all_inputs_);
  all_inputs_ = nullptr;
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Stack, LiteKernelCreator<StackBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Stack, LiteKernelCreator<StackBaseCPUKernel>)
}  // namespace mindspore::kernel
