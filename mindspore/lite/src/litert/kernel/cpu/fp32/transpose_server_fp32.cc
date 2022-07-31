#ifdef BFC_MEMORY
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
#include "src/litert/kernel/cpu/fp32/transpose_server_fp32.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/fp32/pack_fp32.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Transpose;

namespace mindspore::kernel {
namespace {
constexpr int64_t kMinCostPerThread = 1 << 18;
}
int TransposeServerCPUKernel::ReSize() {
  auto ret = TransposeBaseCPUKernel::ReSize();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Do transpose resize failed.";
    return ret;
  }
  if (!is_valid_ || opt_run_) {
    return RET_OK;
  }
  ComputeIndividualOfflineInfo();
  return ChooseThreadCuttingStrategy();
}

void TransposeServerCPUKernel::ComputeIndividualOfflineInfo() {
  MS_ASSERT(param_->num_axes_ >= C3NUM);
  overflow_points_.resize(param_->num_axes_);
  for (int i = 0; i < param_->num_axes_; ++i) {
    overflow_points_[i] = (out_shape_[i] - 1);
  }
  strides_.resize(param_->num_axes_);
  for (int i = 0; i < param_->num_axes_; ++i) {
    strides_[i] = param_->strides_[param_->perm_[i]];
  }
  std::vector<int64_t> in_strides_temp = strides_;
  for (int i = param_->num_axes_ - C2NUM; i >= 0; --i) {
    strides_[i] =
      strides_[i] - in_strides_temp[i + 1] - in_strides_temp[i + 1] * overflow_points_[i + 1] + strides_[i + 1];
  }
}

int TransposeServerCPUKernel::ChooseThreadCuttingStrategy() {
  block_boundary_infos_.clear();
  int64_t element_num = in_tensors_.front()->ElementsNum();
  if (element_num <= kMinCostPerThread) {
    thread_num_ = 1;
  } else {
    thread_num_ = MSMIN(op_parameter_->thread_num_, UP_DIV(element_num, kMinCostPerThread));
  }
  if (thread_num_ < 1) {
    thread_num_ = 1;
  }
  if (thread_num_ > C4NUM) {
    thread_num_ = C4NUM;
  }
  int64_t block_size = element_num / thread_num_;
  int64_t remain_data = element_num - block_size * thread_num_;
  int64_t split_point = 0;
  block_boundary_infos_.clear();
  std::vector<int64_t> post_multi(param_->num_axes_, 1);
  for (int i = param_->num_axes_ - C2NUM; i >= 0; --i) {
    post_multi[i] = post_multi[i + 1] * out_shape_[i + 1];
  }
  while (split_point < element_num) {
    TransposeBlockBoundaryInfo block_boundary_info;
    int64_t in_offset = 0;
    block_boundary_info.out_start_offset = split_point;
    for (int i = 0; i < param_->num_axes_; ++i) {
      block_boundary_info.start_dim[i] = split_point / post_multi[i] % out_shape_[i];
      in_offset += block_boundary_info.start_dim[i] * param_->strides_[param_->perm_[i]];
    }
    block_boundary_info.in_offsets[0] = in_offset;
    split_point += block_size;
    if (remain_data > 0) {
      ++split_point;
      --remain_data;
    }
    if (split_point > element_num) {
      split_point = element_num;
    }
    int64_t size = split_point - block_boundary_info.out_start_offset;
    int last_axis_index = param_->num_axes_ - 1;
    block_boundary_info.sizes[0] =
      MSMIN(size, out_shape_[last_axis_index] - block_boundary_info.start_dim[last_axis_index]);
    size -= block_boundary_info.sizes[0];
    block_boundary_info.sizes[1] = DOWN_ROUND(size, out_shape_[last_axis_index]);
    block_boundary_info.sizes[C2NUM] = size - block_boundary_info.sizes[1];
    int64_t out_offset = block_boundary_info.out_start_offset + block_boundary_info.sizes[0];
    in_offset = 0;
    for (int i = 0; i < param_->num_axes_; ++i) {
      block_boundary_info.start_dim[i] = out_offset / post_multi[i] % out_shape_[i];
      in_offset += block_boundary_info.start_dim[i] * param_->strides_[param_->perm_[i]];
    }
    block_boundary_info.in_offsets[1] = in_offset;
    block_boundary_infos_.push_back(block_boundary_info);
  }
  thread_num_ = block_boundary_infos_.size();
  return RET_OK;
}

int TransposeServerCPUKernel::DoTransposeSingleThread() { return DoTransposeMultiThread(0); }

int TransposeServerCPUKernel::DoTransposeMultiThread(int task_id) {
  if (opt_run_) {
    PackNHWCToNCHWFp32(in_data_, out_data_, opt_param_[FIRST_INPUT], opt_param_[SECOND_INPUT], opt_param_[THIRD_INPUT],
                       task_id, thread_num_);
    return RET_OK;
  }
  DoTransposeServer(static_cast<float *>(in_data_), static_cast<float *>(out_data_), overflow_points_.data(),
                    strides_.data(), param_->num_axes_, &block_boundary_infos_[task_id]);
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Transpose, LiteKernelCreator<TransposeServerCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Transpose, LiteKernelCreator<TransposeServerCPUKernel>)
}  // namespace mindspore::kernel
#endif
