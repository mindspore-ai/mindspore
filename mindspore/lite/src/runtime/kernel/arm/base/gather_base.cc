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

#include "src/runtime/kernel/arm/base/gather_base.h"
#include <limits>
#include "nnacl/base/gather_base.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int GatherRun(const void *cdata, int task_id, float, float) {
  auto gather_kernel = reinterpret_cast<const GatherBaseCPUKernel *>(cdata);
  auto error_code = gather_kernel->DoGather(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "GatherRun error task_id[" << task_id << "] error_code[" << error_code << "]";
  }
  return error_code;
}

int GatherBaseCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), kInputSize2);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_.at(THIRD_INPUT));
  CHECK_NULL_RETURN(in_tensors_.at(THIRD_INPUT)->data());
  axis_ = *(reinterpret_cast<int *>(in_tensors_.at(THIRD_INPUT)->data()));
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int GatherBaseCPUKernel::ReSize() { return ChooseThreadCuttingstrategy(); }

int GatherBaseCPUKernel::DoGather(int task_id) const {
  auto *int8_in = reinterpret_cast<int8_t *>(in_tensors_[FIRST_INPUT]->data());
  CHECK_NULL_RETURN(int8_in);
  auto in_shape = in_tensors_[FIRST_INPUT]->shape();
  MS_CHECK_LT(axis_, static_cast<int>(in_shape.size()), RET_ERROR);
  const int64_t limit = in_shape.at(axis_);
  auto *int8_out = reinterpret_cast<int8_t *>(out_tensors_[FIRST_INPUT]->data());
  CHECK_NULL_RETURN(int8_out);
  int data_size = static_cast<int>(lite::DataTypeSize(in_tensors_[FIRST_INPUT]->data_type()));
  auto index_num = in_tensors_[SECOND_INPUT]->ElementsNum();
  int64_t byte_inner_size = inner_size_ * data_size;
  int64_t byte_out_stride = index_num * byte_inner_size;
  int64_t all_count = split_by_index ? index_num : outer_size_;
  int64_t count = (task_id < static_cast<int>(split_points_.size()) - 1)
                    ? split_points_[task_id + 1] - split_points_[task_id]
                    : all_count - split_points_[task_id];

  int ret = RET_OK;
  if (split_by_index) {
    int *indices_data = indices_data_ + split_points_[task_id];
    int8_out += split_points_[task_id] * byte_inner_size;
    ret = Gather(int8_in, outer_size_, byte_inner_size, limit, indices_data, count, int8_out, byte_out_stride);
  } else {
    int8_in += split_points_[task_id] * limit * byte_inner_size;
    int8_out += split_points_[task_id] * byte_out_stride;
    ret = Gather(int8_in, count, byte_inner_size, limit, indices_data_, index_num, int8_out, byte_out_stride);
  }
  return ret;
}

int GatherBaseCPUKernel::Run() {
  bool isIndicesInt32 = in_tensors_[SECOND_INPUT]->data_type() == kNumberTypeInt32;
  int ret = AssignIndicesData(isIndicesInt32);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "AssignIndicesData failed, error_code[" << ret << "]";
    return ret;
  }

  ret = ParallelLaunch(this->ms_context_, GatherRun, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Gather function error error_code[" << ret << "]";
  }
  if (!isIndicesInt32) {
    ms_context_->allocator->Free(indices_data_);
    indices_data_ = nullptr;
  }
  return ret;
}

int GatherBaseCPUKernel::ChooseThreadCuttingstrategy() {
  auto in_shape = in_tensors_[FIRST_INPUT]->shape();
  int in_rank = static_cast<int>(in_shape.size());
  MS_CHECK_TRUE_MSG(axis_ < in_rank, RET_ERROR, "gather's inputs are invalid.");
  outer_size_ = 1;
  for (int i = 0; i < axis_; ++i) {
    outer_size_ *= in_shape.at(i);
  }
  inner_size_ = 1;
  for (int i = axis_ + 1; i < in_rank; ++i) {
    inner_size_ *= in_shape.at(i);
  }
  int64_t all_count = outer_size_;
  auto index_num = in_tensors_[SECOND_INPUT]->ElementsNum();
  if (outer_size_ >= index_num) {
    split_by_index = false;
  } else {
    all_count = index_num;
    split_by_index = true;
  }
  int64_t count_step = MSMAX(all_count / op_parameter_->thread_num_, 1);
  int64_t count_remaining = MSMAX(all_count - count_step * op_parameter_->thread_num_, 0);
  split_points_.clear();
  int64_t split_point = 0;
  while (split_point < all_count) {
    split_points_.push_back(split_point);
    split_point += count_step;
    if (count_remaining > 0) {
      split_point += 1;
      --count_remaining;
    }
  }
  thread_count_ = static_cast<int>(split_points_.size());
  return RET_OK;
}
}  // namespace mindspore::kernel
