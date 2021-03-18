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
#include "src/runtime/kernel/arm/base/strided_slice.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_StridedSlice;

namespace mindspore::kernel {
int StridedSliceCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

void StridedSliceCPUKernel::InitFastRunParam() {
  auto in_shape = in_tensors_.front()->shape();
  auto out_shape = out_tensors_.front()->shape();
  // reset && cal inner, outer
  outer_ = 1;
  inner_ = 1;
  for (int i = 0; i < split_axis_; ++i) {
    outer_ *= in_shape[i];
  }
  for (size_t i = split_axis_ + 1; i < in_shape.size(); i++) {
    inner_ *= in_shape[i];
  }
  // decide multi-thread launch strategy
  if (outer_ == 1) {
    parallel_on_split_axis_ = true;
    cal_num_per_thread_ = UP_DIV(out_shape[split_axis_], context_->thread_num_);
  } else {
    parallel_on_outer_ = true;
    cal_num_per_thread_ = UP_DIV(outer_, context_->thread_num_);
  }
}

int StridedSliceCPUKernel::ReSize() {
  fast_run_ = MatchFastPattern();
  if (fast_run_) {
    InitFastRunParam();
  }
  return RET_OK;
}

bool StridedSliceCPUKernel::MatchFastPattern() {
  // This function is seeking if that the number of only one dimension
  // is different between input and output. If so, we can do some trick.
  // Example 1:
  // input shape info:  [1, 80, 46, 40]
  // output shape info: [1, 80, 20, 40]
  // Example 2:
  // input shape info:  [1, 46, 40]
  // output shape info: [1, 20, 40]
  auto in_shape = in_tensors_.front()->shape();
  auto out_shape = out_tensors_.front()->shape();
  if (in_shape.size() != out_shape.size()) {
    return false;
  }
  std::vector<int> axis_list;
  for (size_t i = 0; i < in_shape.size(); ++i) {
    if (in_shape[i] != out_shape[i]) {
      axis_list.emplace_back(i);
    }
  }
  if (axis_list.size() == 1) {
    split_axis_ = axis_list.front();
    return true;
  }
  return false;
}

int StridedSliceCPUKernel::FastRunImpl(int task_id) {
  auto in_shape = in_tensors_.front()->shape();
  auto out_shape = out_tensors_.front()->shape();
  int begin_index = param_->begins_[split_axis_];
  int caled_num = task_id * cal_num_per_thread_;
  if (parallel_on_outer_) {
    uint8_t *cur_in_ptr = input_ptr_ + (caled_num * in_shape[split_axis_] + begin_index) * inner_size_;
    uint8_t *cur_out_ptr = output_ptr_ + caled_num * out_shape[split_axis_] * inner_size_;
    int cur_outer = outer_ - caled_num;
    if (cur_outer <= 0) {
      return RET_OK;
    }
    if (cur_outer > cal_num_per_thread_) {
      cur_outer = cal_num_per_thread_;
    }
    FastStride(cur_in_ptr, cur_out_ptr, out_shape[split_axis_], param_->strides_[split_axis_], cur_outer, inner_size_,
               in_shape[split_axis_] * inner_size_);
  } else {
    MS_ASSERT(parallel_on_split_axis_);
    uint8_t *cur_in_ptr = input_ptr_ + (caled_num * param_->strides_[split_axis_] + begin_index) * inner_size_;
    uint8_t *cur_out_ptr = output_ptr_ + caled_num * inner_size_;
    int cal_axis_num = out_shape[split_axis_] - caled_num;
    if (cal_axis_num <= 0) {
      return RET_OK;
    }
    if (cal_axis_num > cal_num_per_thread_) {
      cal_axis_num = cal_num_per_thread_;
    }
    FastStride(cur_in_ptr, cur_out_ptr, cal_axis_num, param_->strides_[split_axis_], 1, inner_size_, 0);
  }
  return RET_OK;
}

int StrideRun(void *cdata, int task_id) {
  auto stride = reinterpret_cast<StridedSliceCPUKernel *>(cdata);
  auto ret = stride->FastRunImpl(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "StrideRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int StridedSliceCPUKernel::FastRun() {
  // Update length of inner size, because data type of tensor may be changed
  // from float32 to float16 during fp16 sub-graph partition process.
  auto input = in_tensors_.front();
  switch (input->data_type()) {
    case kNumberTypeInt8:
      inner_size_ = inner_ * sizeof(int8_t);
      break;
    case kNumberTypeFloat32:
      inner_size_ = inner_ * sizeof(float);
      break;
    case kNumberTypeFloat16:
      inner_size_ = inner_ * sizeof(int16_t);
      break;
    case kNumberTypeInt32:
      inner_size_ = inner_ * sizeof(int32_t);
      break;
    default:
      MS_LOG(ERROR) << "Not supported data type: " << input->data_type();
      return RET_ERROR;
  }
  input_ptr_ = reinterpret_cast<uint8_t *>(in_tensors_.front()->data_c());
  output_ptr_ = reinterpret_cast<uint8_t *>(out_tensors_.front()->data_c());
  auto ret = ParallelLaunch(this->context_->thread_pool_, StrideRun, this, context_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Stride run error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int StridedSliceCPUKernel::NormalRun() {
  auto input = in_tensors_.at(0);
  MS_ASSERT(input);
  switch (input->data_type()) {
    case kNumberTypeInt8:
      param_->data_type = kDataTypeInt8;
      break;
    case kNumberTypeFloat32:
      param_->data_type = kDataTypeFloat;
      break;
    case kNumberTypeFloat16:
      param_->data_type = kDataTypeFloat16;
      break;
    case kNumberTypeInt32:
      param_->data_type = kDataTypeInt;
      break;
    default:
      MS_LOG(ERROR) << "Not supported data type: " << input->data_type();
      return RET_ERROR;
  }
  auto output = out_tensors_.at(0);
  MS_ASSERT(output);
  auto ret = DoStridedSlice(input->data_c(), output->data_c(), param_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "StridedSlice error error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int StridedSliceCPUKernel::Run() {
  if (fast_run_) {
    return FastRun();
  }
  return NormalRun();
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_StridedSlice, LiteKernelCreator<StridedSliceCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_StridedSlice, LiteKernelCreator<StridedSliceCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_StridedSlice, LiteKernelCreator<StridedSliceCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_StridedSlice, LiteKernelCreator<StridedSliceCPUKernel>)
}  // namespace mindspore::kernel
