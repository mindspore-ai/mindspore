/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "src/runtime/kernel/arm/base/split_with_over_lap_base.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/tensor.h"
#include "nnacl/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SplitWithOverlap;
#define MIN_NUM_SPLIT 2

namespace mindspore::kernel {

int SplitWithOverlapBaseCPUKernel::CalculateSplitedShapes(const std::vector<int> &shape) {
  int total_block_count = 0;
  CHECK_LESS_RETURN(SPLIT_MAX_SLICE_NUM, param_->num_split_ + 1);
  for (auto i = 0; i < param_->num_split_; i++) {
    total_block_count += param_->ratio_[i];
  }
  CHECK_LESS_RETURN(static_cast<int>(shape.size()), param_->split_dim_ + 1);
  auto split_dim_size = shape[param_->split_dim_];

  std::vector<int> borders;
  borders.emplace_back(0);
  int visited_block = 0;
  for (auto i = 0; i < param_->num_split_ - 1; i++) {
    visited_block += param_->ratio_[i];
    MS_CHECK_FALSE(INT_MUL_OVERFLOW(split_dim_size, visited_block), RET_ERROR);
    auto cur_border = UP_DIV(split_dim_size * visited_block, total_block_count);
    borders.emplace_back(cur_border);
  }
  borders.emplace_back(split_dim_size);

  for (auto i = 0; i < param_->num_split_; i++) {
    start_indices_.emplace_back(borders[i]);
    end_indices_.emplace_back(borders[i + 1]);

    // overlap: calibrate start_indices and end_indices by adding extends
    start_indices_[i] -= param_->extend_top_[i];
    end_indices_[i] += param_->extend_bottom_[i];
  }
  return RET_OK;
}

int SplitWithOverlapBaseCPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_LESS_RETURN(param_->num_split_, MIN_NUM_SPLIT);
  return ReSize();
}

int SplitWithOverlapBaseCPUKernel::ReSize() {
  auto in_tensor = in_tensors_.front();
  auto input_shape = in_tensor->shape();

  start_indices_.clear();
  end_indices_.clear();

  if (CalculateSplitedShapes(input_shape) != RET_OK) {
    MS_LOG(ERROR) << "CalculateSplitedShapes error.";
    return RET_ERROR;
  }

  param_->element_bytes_ = static_cast<int>(lite::DataTypeSize(in_tensor->data_type()));

  param_->outer_total_dim_ = 1;
  param_->inner_stride_ = 1;

  for (int i = 0; i < static_cast<int>(input_shape.size()); i++) {
    if (i < param_->split_dim_) {
      param_->outer_total_dim_ *= input_shape[i];
    }
    if (i == param_->split_dim_) {
      param_->split_dim_size_ = input_shape[param_->split_dim_];
    }
    if (i > param_->split_dim_) {
      param_->inner_stride_ *= input_shape[i];
    }
  }

  thread_count_ = MSMIN(param_->num_split_, op_parameter_->thread_num_);
  return RET_OK;
}

int SplitWithOverlapBaseCPUKernel::Split(int task_id) {
  for (int current_slice_task = task_id; current_slice_task < param_->num_split_; current_slice_task += thread_count_) {
    if (DoSplitWithOverlapParallel(input_ptr_, output_ptr_.data(), current_slice_task, param_, start_indices_.data(),
                                   end_indices_.data()) != NNACL_OK) {
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int SplitWithOverlapRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto g_kernel = reinterpret_cast<SplitWithOverlapBaseCPUKernel *>(cdata);
  auto ret = g_kernel->Split(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SplitWithOverlapRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }

  return RET_OK;
}

int SplitWithOverlapBaseCPUKernel::Run() {
  input_ptr_ = reinterpret_cast<char *>(in_tensors_.front()->data());
  if (input_ptr_ == nullptr) {
    return RET_NULL_PTR;
  }
  output_ptr_.clear();
  CHECK_LESS_RETURN(static_cast<int>(out_tensors_.size()), param_->num_split_);
  for (int i = 0; i < param_->num_split_; i++) {
    output_ptr_.push_back(reinterpret_cast<char *>(out_tensors_.at(i)->data()));
    if (output_ptr_.at(i) == nullptr) {
      return RET_NULL_PTR;
    }
  }

  auto ret = ParallelLaunch(this->ms_context_, SplitWithOverlapRun, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ParallelLaunch for SplitWIthOverlapRun run fail. errorcode:[" << ret << "]";
    return RET_ERROR;
  }

  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SplitWithOverlap, LiteKernelCreator<SplitWithOverlapBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_SplitWithOverlap, LiteKernelCreator<SplitWithOverlapBaseCPUKernel>)
}  // namespace mindspore::kernel
