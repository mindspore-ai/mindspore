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
#include "src/runtime/kernel/arm/base/split_base.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Split;

namespace mindspore::kernel {
int SplitBaseCPUKernel::Init() {
  auto split_dim = param->split_dim_;
  param->split_dim_ = split_dim >= 0 ? split_dim : in_tensors_.front()->shape().size() + split_dim;

  output_ptr_.resize(param->num_split_);
  for (size_t i = 0; i < output_ptr_.size(); i++) {
    output_ptr_.at(i) = nullptr;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int SplitBaseCPUKernel::ReSize() {
  auto in_tensor = in_tensors_.front();
  auto input_shape = in_tensor->shape();

  MS_ASSERT(param);
  MS_ASSERT(input_shape.size() >= 1 && input_shape.size() <= SPLIT_STRIDES_SIZE);
  auto split_dim = param->split_dim_;
  param->split_dim_ = split_dim >= 0 ? split_dim : in_tensors_.front()->shape().size() + split_dim;
  param->strides_[input_shape.size() - 1] = 1;
  for (int i = input_shape.size() - 2; i >= 0; i--) {
    param->strides_[i] = param->strides_[i + 1] * input_shape.at(i + 1);
  }

  MS_ASSERT(static_cast<size_t>(param->split_dim_) < input_shape.size());
  param->split_count_ =
    param->strides_[0] * input_shape.at(0) / (input_shape.at(param->split_dim_) * param->strides_[param->split_dim_]);
  param->n_dims_ = input_shape.size();

  if (param->split_sizes_[0] == 0) {
    MS_ASSERT(param->num_split_ > 0 && static_cast<int>(param->num_split_) <= input_shape[param->split_dim_]);
    if (input_shape[param->split_dim_] % param->num_split_ != 0) {
      MS_LOG(ERROR) << "Default split size is not usable.";
      return RET_ERROR;
    }
    int split_size = input_shape.at(param->split_dim_) / param->num_split_;
    for (int i = 0; i < param->num_split_; i++) {
      param->split_sizes_[i] = split_size;
    }
  }

  if (param->split_sizes_[param->num_split_ - 1] == -1) {
    int split_shape_end = input_shape.at(param->split_dim_);
    for (int i = 0; i < param->num_split_ - 1; i++) {
      split_shape_end -= param->split_sizes_[i];
    }
    param->split_sizes_[param->num_split_ - 1] = split_shape_end;
  }

  num_unit_ = param->split_count_ * param->num_split_;
  thread_n_num_ = MSMIN(op_parameter_->thread_num_, num_unit_);
  if (thread_n_num_ != 0) {
    thread_n_stride_ = UP_DIV(num_unit_, thread_n_num_);
  }
  return RET_OK;
}

int SplitBaseCPUKernel::Split(int task_id) {
  int num_unit_thread = MSMIN(thread_n_stride_, num_unit_ - task_id * thread_n_stride_);
  if (num_unit_thread <= 0) {
    return RET_OK;
  }
  int thread_offset = task_id * thread_n_stride_;
  auto ret = DoSplit(input_ptr_, output_ptr_.data(), in_tensors_.front()->shape().data(), thread_offset,
                     num_unit_thread, param, lite::DataTypeSize(in_tensors_.front()->data_type()));
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Split error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

static int SplitRun(void *cdata, int task_id) {
  auto g_kernel = reinterpret_cast<SplitBaseCPUKernel *>(cdata);
  auto ret = g_kernel->Split(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SplitRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SplitBaseCPUKernel::Run() {
  auto input_tensor = in_tensors_.at(0);
  input_ptr_ = input_tensor->data_c();

  for (int i = 0; i < param->num_split_; i++) {
    auto output_tensor = out_tensors_.at(i);
    output_ptr_.at(i) = output_tensor->data_c();
  }

  auto ret = ParallelLaunch(this->context_->thread_pool_, SplitRun, this, thread_n_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "split error error_code[" << ret << "]";
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Split, LiteKernelCreator<SplitBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Split, LiteKernelCreator<SplitBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Split, LiteKernelCreator<SplitBaseCPUKernel>)
}  // namespace mindspore::kernel
