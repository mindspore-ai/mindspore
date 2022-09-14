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

#include "src/litert/kernel/cpu/base/concat_base.h"
#include <algorithm>
#include <utility>
#include <vector>
#include "src/litert/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Concat;

namespace mindspore::kernel {
namespace {
constexpr int kMinCostPerThread = 16384;
}
int ConcatBaseRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto concat_kernel = reinterpret_cast<ConcatBaseCPUKernel *>(cdata);
  CHECK_NULL_RETURN(concat_kernel);
  auto error_code = concat_kernel->DoConcat(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ConcatRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConcatBaseCPUKernel::DoConcat(int task_id) {
  auto all_bytes = static_cast<int64_t>(out_tensors_.front()->Size());
  int64_t start = block_splits_[task_id];
  int64_t end = task_id < (static_cast<int>(block_splits_.size()) - 1) ? block_splits_[task_id + 1] : all_bytes;
  int64_t start_row = start / inner_sizes_.back();
  int64_t end_row = end / inner_sizes_.back();
  std::vector<const uint8_t *> src;
  for (size_t i = 0; i < inputs_ptr_.size(); ++i) {
    src.push_back(inputs_ptr_[i] + start_row * inner_sizes_[i]);
  }
  uint8_t *out = output_ + start;
  int input_index = block_boundary_infos_[task_id].begin_input;
  int end_index = block_boundary_infos_[task_id].end_input;
  if (start_row == end_row) {
    if (input_index == end_index) {
      memcpy(out, src[input_index] + block_boundary_infos_[task_id].begin_point,
             block_boundary_infos_[task_id].end_point - block_boundary_infos_[task_id].begin_point);
      return RET_OK;
    }
    int64_t size = inner_sizes_[input_index] - block_boundary_infos_[task_id].begin_point;
    memcpy(out, src[input_index] + block_boundary_infos_[task_id].begin_point, size);
    out += size;
    ++input_index;
    for (; input_index < end_index; ++input_index) {
      memcpy(out, src[input_index], inner_sizes_[input_index]);
      out += inner_sizes_[input_index];
    }
    memcpy(out, src[input_index], block_boundary_infos_[task_id].end_point);
    return RET_OK;
  }
  for (int i = 0; i < input_index; ++i) {
    src[i] += inner_sizes_[i];
  }
  int64_t size = inner_sizes_[input_index] - block_boundary_infos_[task_id].begin_point;
  memcpy(out, src[input_index] + block_boundary_infos_[task_id].begin_point, size);
  src[input_index] += inner_sizes_[input_index];
  out += size;
  ++input_index;
  for (; input_index < static_cast<int>(inputs_ptr_.size()); ++input_index) {
    memcpy(out, src[input_index], inner_sizes_[input_index]);
    src[input_index] += inner_sizes_[input_index];
    out += inner_sizes_[input_index];
  }
  ++start_row;
  for (; start_row < end_row; ++start_row) {
    for (input_index = 0; input_index < static_cast<int>(inputs_ptr_.size()); ++input_index) {
      memcpy(out, src[input_index], inner_sizes_[input_index]);
      src[input_index] += inner_sizes_[input_index];
      out += inner_sizes_[input_index];
    }
  }
  for (input_index = 0; input_index < end_index; ++input_index) {
    memcpy(out, src[input_index], inner_sizes_[input_index]);
    out += inner_sizes_[input_index];
  }
  memcpy(out, src[end_index], block_boundary_infos_[task_id].end_point);
  return RET_OK;
}

int ConcatBaseCPUKernel::Prepare() {
  MS_CHECK_TRUE_RET(!in_tensors_.empty(), RET_ERROR);
  for (auto in_tensor : in_tensors_) {
    CHECK_NULL_RETURN(in_tensor);
  }
  MS_CHECK_TRUE_RET(out_tensors_.size() == 1, RET_ERROR);
  CHECK_NULL_RETURN(out_tensors_.front());
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ConcatBaseCPUKernel::ReSize() {
  concat_param_->axis_ = concat_param_->axis_ >= 0
                           ? concat_param_->axis_
                           : static_cast<int>(in_tensors_.front()->shape().size()) + concat_param_->axis_;
  MS_CHECK_TRUE_MSG(
    concat_param_->axis_ >= 0 && concat_param_->axis_ < static_cast<int>(in_tensors_.front()->shape().size()),
    RET_ERROR, "concat-axis is invalid.");
  auto ret = InitDynamicStatus();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "update dynamic-status failed.";
    return ret;
  }
  if (outer_size_ == 0 || inner_sizes_.back() == 0) {
    return RET_OK;
  }
  ret = ChooseThreadCuttingStrategy();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "decide thread cutting strategy failed.";
    return ret;
  }
  return RET_OK;
}

int ConcatBaseCPUKernel::InitDynamicStatus() {
  inner_sizes_.clear();
  is_with_data_.clear();
  int64_t output_inner_size = 0;
  for (size_t i = 0; i < in_tensors_.size(); ++i) {
    auto shape = in_tensors_[i]->shape();
    MS_CHECK_TRUE_MSG(concat_param_->axis_ < static_cast<int>(shape.size()), RET_ERROR, "concat-axis is invalid.");
    int64_t outer_size = 1;
    for (int j = 0; j < concat_param_->axis_; ++j) {
      outer_size *= shape[j];
    }
    auto inner_size = data_size_;
    if (out_tensors_.front()->data_type() == kNumberTypeBool) {
      inner_size = sizeof(bool);
    }

    MS_CHECK_TRUE_MSG(inner_size > 0, RET_ERROR, "data-type is invalid.");
    for (int j = concat_param_->axis_; j < static_cast<int>(shape.size()); ++j) {
      inner_size *= shape[j];
    }
    if (i == 0) {
      outer_size_ = outer_size;
    } else {
      MS_CHECK_TRUE_MSG(outer_size_ == outer_size, RET_ERROR, "input tensor is invalid.");
    }
    if (inner_size == 0) {
      is_with_data_.push_back(false);
      continue;
    }
    is_with_data_.push_back(true);
    inner_sizes_.push_back(inner_size);
    output_inner_size += inner_size;
  }
  inner_sizes_.push_back(output_inner_size);
  return RET_OK;
}

int ConcatBaseCPUKernel::ChooseThreadCuttingStrategy() {
  block_splits_.clear();
  block_boundary_infos_.clear();
  MS_CHECK_TRUE_MSG(op_parameter_->thread_num_ > 0, RET_ERROR, "available thread is 0.");
  auto all_bytes = static_cast<int64_t>(out_tensors_.front()->Size());
  int64_t thread_count = std::max(
    static_cast<int64_t>(1), std::min(all_bytes / kMinCostPerThread, static_cast<int64_t>(op_parameter_->thread_num_)));
  int64_t block_size = all_bytes / thread_count;
  int64_t remain_byte = all_bytes - block_size * thread_count;
  std::vector<int64_t> pre_sum;
  int64_t init_sum = 0;
  MS_CHECK_TRUE_MSG(!inner_sizes_.empty(), RET_ERROR, "er-size is invalid.");
  for (size_t i = 0; i < inner_sizes_.size() - 1; ++i) {
    init_sum += inner_sizes_[i];
    pre_sum.push_back(init_sum);
  }
  auto ComputeBoundaryFunc = [this, &pre_sum](int64_t offset) {
    size_t index = 0;
    for (; index < pre_sum.size(); ++index) {
      if (offset < pre_sum[index]) {
        break;
      }
    }
    return std::make_pair(index, inner_sizes_[index] - (pre_sum[index] - offset));
  };
  int64_t block_spilt = 0;
  while (block_spilt < all_bytes) {
    block_splits_.push_back(block_spilt);
    block_spilt += block_size;
    if (remain_byte > 0) {
      ++block_spilt;
      --remain_byte;
    }
    int64_t start = block_splits_.back();
    int64_t end = block_spilt > all_bytes ? all_bytes : block_spilt;
    int64_t start_offset = start - DOWN_ROUND(start, inner_sizes_.back());
    int64_t end_offset = end - DOWN_ROUND(end, inner_sizes_.back());
    BlockBoundaryInfo block_boundary_info;
    auto boundary_info = ComputeBoundaryFunc(start_offset);
    block_boundary_info.begin_input = boundary_info.first;
    block_boundary_info.begin_point = boundary_info.second;
    boundary_info = ComputeBoundaryFunc(end_offset);
    block_boundary_info.end_input = boundary_info.first;
    block_boundary_info.end_point = boundary_info.second;
    block_boundary_infos_.push_back(block_boundary_info);
  }
  return RET_OK;
}

int ConcatBaseCPUKernel::Run() {
  if (outer_size_ == 0 || inner_sizes_.back() == 0) {
    return RET_OK;
  }
  inputs_ptr_.clear();
  for (size_t i = 0; i < in_tensors_.size(); ++i) {
    if (!is_with_data_[i]) {
      continue;
    }
    MS_CHECK_TRUE_MSG(in_tensors_[i]->data() != nullptr, RET_ERROR, "input tensor data is nullptr.");
    inputs_ptr_.push_back(static_cast<const uint8_t *>(in_tensors_[i]->data()));
  }
  output_ = static_cast<uint8_t *>(out_tensors_.front()->data());
  MS_CHECK_TRUE_MSG(output_ != nullptr, RET_ERROR, "output data is a nullptr.");
  auto ret = ParallelLaunch(this->ms_context_, ConcatBaseRun, this, block_splits_.size());
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Concat, LiteKernelCreator<ConcatBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Concat, LiteKernelCreator<ConcatBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_Concat, LiteKernelCreator<ConcatBaseCPUKernel>)
}  // namespace mindspore::kernel
