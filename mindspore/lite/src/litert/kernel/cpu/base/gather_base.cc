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

#include "src/litert/kernel/cpu/base/gather_base.h"
#include <algorithm>

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Gather;

namespace mindspore::kernel {
int GatherRun(void *cdata, int task_id, float, float) {
  auto gather_kernel = reinterpret_cast<const GatherBaseCPUKernel *>(cdata);
  CHECK_NULL_RETURN(gather_kernel);
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

int GatherBaseCPUKernel::ReSize() {
  auto status = InitDynamicStatus();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Gather init status failed when resizing." << name_;
    return status;
  }
  return ChooseThreadCuttingStrategy();
}

int GatherBaseCPUKernel::DoGather(int task_id) const {
  if (task_id < 0 || static_cast<size_t>(task_id) >= block_boundary_infos_.size()) {
    MS_LOG(ERROR) << "task_id " << task_id << " is out of range, node is " << name_;
    return RET_ERROR;
  }
  auto *int8_in = reinterpret_cast<int8_t *>(in_tensors_[FIRST_INPUT]->data());
  CHECK_NULL_RETURN(int8_in);
  auto *int8_out = reinterpret_cast<int8_t *>(out_tensors_[FIRST_INPUT]->data());
  CHECK_NULL_RETURN(int8_out);
  auto begin_batch = block_boundary_infos_[task_id].begin_batch;
  auto begin_index = block_boundary_infos_[task_id].begin_index;
  auto end_batch = block_boundary_infos_[task_id].end_batch;
  auto end_index = block_boundary_infos_[task_id].end_index;
  int64_t byte_in_stride = limit_ * byte_inner_size_;
  int8_in += begin_batch * byte_in_stride;
  int8_out += begin_batch * indices_size_ * byte_inner_size_ + begin_index * byte_inner_size_;
  auto HandleCopy = [this, int8_out, int8_in, byte_in_stride](int64_t begin, int64_t end) mutable {
    for (; begin < end; ++begin) {
      int index = indices_data_[begin];
      index = (index < 0 ? index + limit_ : index);
      if (index < 0 || index >= limit_) {
        memset(int8_out, 0, byte_inner_size_);
      } else {
        memcpy(int8_out, int8_in + index * byte_inner_size_, byte_inner_size_);
      }
      int8_out += byte_inner_size_;
    }
    int8_in += byte_in_stride;
  };
  if (begin_batch == end_batch) {
    HandleCopy(begin_index, end_index);
    return RET_OK;
  }
  HandleCopy(begin_index, indices_size_);
  ++begin_batch;
  for (; begin_batch < end_batch; ++begin_batch) {
    HandleCopy(0, indices_size_);
  }
  HandleCopy(0, end_index);
  return RET_OK;
}

int GatherBaseCPUKernel::Run() {
  if (outer_size_ == 0 || indices_size_ == 0 || byte_inner_size_ == 0) {
    return RET_OK;
  }
  bool isIndicesInt32 = in_tensors_[SECOND_INPUT]->data_type() == kNumberTypeInt32;
  int ret = AssignIndicesData(isIndicesInt32);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "AssignIndicesData failed, error_code[" << ret << "]";
    return ret;
  }

  ret = ParallelLaunch(this->ms_context_, GatherRun, this, block_boundary_infos_.size());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Gather function error error_code[" << ret << "]";
  }
  if (!isIndicesInt32) {
    ms_context_->allocator->Free(indices_data_);
    indices_data_ = nullptr;
  }
  return ret;
}

int GatherBaseCPUKernel::InitDynamicStatus() {
  auto in_shape = in_tensors_[FIRST_INPUT]->shape();
  int in_rank = static_cast<int>(in_shape.size());
  MS_CHECK_TRUE_MSG(axis_ >= 0 && axis_ < in_rank, RET_ERROR, "gather's inputs are invalid.");
  limit_ = in_shape[axis_];
  outer_size_ = 1;
  for (int i = 0; i < axis_; ++i) {
    outer_size_ *= in_shape.at(i);
  }
  byte_inner_size_ = static_cast<int64_t>(lite::DataTypeSize(out_tensors_.front()->data_type()));
  for (int i = axis_ + 1; i < in_rank; ++i) {
    byte_inner_size_ *= in_shape.at(i);
  }
  indices_size_ = in_tensors_[SECOND_INPUT]->ElementsNum();
  return RET_OK;
}

int GatherBaseCPUKernel::UpdateThreadNumProcess(int32_t kernel_type, int64_t per_unit_load_num,
                                                int64_t per_unit_store_num, int64_t unit_num) {
  auto all_bytes = static_cast<int64_t>(out_tensors_.front()->Size());
  constexpr int kMinCostPerThread = 16384;
  if (all_bytes <= static_cast<int64_t>(kMinCostPerThread)) {
    thread_num_ = 1;
    return RET_OK;
  }

  thread_num_ =
    lite::UpdateThreadNum(kernel_type, per_unit_load_num, per_unit_store_num, unit_num, op_parameter_->thread_num_);
  return lite::RET_OK;
}

int GatherBaseCPUKernel::ChooseThreadCuttingStrategy() {
  block_boundary_infos_.clear();
  if (outer_size_ == 0 || indices_size_ == 0 || byte_inner_size_ == 0) {
    return RET_OK;
  }

  if (UpdateThreadNumPass(TC_PTYPE(PrimitiveType_Gather), 0, byte_inner_size_, out_tensors_.front()->Size()) !=
      RET_OK) {
    return RET_ERROR;
  }
  if (thread_num_ == 1) {
    block_boundary_infos_.emplace_back(BlockBoundaryInfo{0, 0, outer_size_, 0});
    return RET_OK;
  }
  auto total_block = outer_size_ * indices_size_;
  int64_t block_size = total_block / thread_num_;
  auto remain_block = total_block - block_size * thread_num_;
  int64_t start = 0;
  while (start < total_block) {
    BlockBoundaryInfo block_boundary_info{};
    block_boundary_info.begin_batch = start / indices_size_;
    block_boundary_info.begin_index = start % indices_size_;
    start += block_size;
    if (remain_block > 0) {
      ++start;
      --remain_block;
    }
    if (start >= total_block) {
      start = total_block;
    }
    block_boundary_info.end_batch = start / indices_size_;
    block_boundary_info.end_index = start % indices_size_;
    block_boundary_infos_.push_back(block_boundary_info);
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
