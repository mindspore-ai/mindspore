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
#include "src/litert/kernel/cpu/base/strided_slice.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_StridedSlice;

namespace {
constexpr int kNumInputSize = 2;
constexpr int kNumOutputSize = 1;
}  // namespace
namespace mindspore::kernel {
int StridedSliceCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), kNumInputSize);
  CHECK_LESS_RETURN(out_tensors_.size(), kNumOutputSize);
  CHECK_NULL_RETURN(param_);
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
  for (size_t i = static_cast<size_t>(split_axis_ + 1); i < in_shape.size(); i++) {
    inner_ *= in_shape[i];
  }
  parallel_on_split_axis_ = false;
  parallel_on_outer_ = false;
  outer_ == 1 ? (parallel_on_split_axis_ = true) : (parallel_on_outer_ = true);

  if (UpdateThreadNumPass(TC_TYPE(PrimitiveType_StridedSlice, parallel_on_outer_), 1, 1,
                          out_tensors_.at(0)->ElementsNum()) != RET_OK) {
    MS_LOG(ERROR) << "thread num update thread failed.";
    return;
  }

  cal_num_per_thread_ =
    parallel_on_split_axis_ ? UP_DIV(out_shape[split_axis_], thread_num_) : UP_DIV(outer_, thread_num_);
}

int StridedSliceCPUKernel::ReSize() {
  auto input_tensor = in_tensors_.at(0);
  CHECK_NULL_RETURN(input_tensor);
  auto begin_tensor = in_tensors_.at(1);
  CHECK_NULL_RETURN(begin_tensor);
  CHECK_NULL_RETURN(out_tensors_[0]);
  if (input_tensor->shape().size() > DIMENSION_8D || begin_tensor->shape().size() > DIMENSION_8D) {
    MS_LOG(ERROR) << "StridedSlice not support input rank or begin num exceeds " << DIMENSION_8D;
    return RET_ERROR;
  }
  soft_copy_mode_ = MatchInOutShapeEqualPattern();
  fast_run_ = MatchFastPattern();
  if (fast_run_) {
    InitFastRunParam();
  }

  return RET_OK;
}

bool StridedSliceCPUKernel::MatchInOutShapeEqualPattern() {
  for (int i = 0; i < MAX_SHAPE_SIZE; i++) {
    if (param_->strides_[i] < 0) {
      return false;
    }
  }

  auto in_shape = in_tensors_.front()->shape();
  auto out_shape = out_tensors_.front()->shape();
  if (in_tensors_.front()->data_type() != out_tensors_.front()->data_type()) {
    return false;
  }
  if (in_shape.size() != out_shape.size() || in_shape.size() < 1) {
    return false;
  }
  for (size_t i = 0; i < in_shape.size(); ++i) {
    if (in_shape[i] != out_shape[i] || in_shape[i] == -1) {
      return false;
    }
  }
  return true;
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
    MS_CHECK_TRUE_MSG(parallel_on_split_axis_ == true, RET_ERROR,
                      "Stride slice op should be parallel on axis or outer size.");
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

int StrideRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
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
  input_ptr_ = reinterpret_cast<uint8_t *>(in_tensors_.front()->data());
  CHECK_NULL_RETURN(input_ptr_);
  output_ptr_ = reinterpret_cast<uint8_t *>(out_tensors_.front()->data());
  CHECK_NULL_RETURN(output_ptr_);
  auto ret = ParallelLaunch(this->ms_context_, StrideRun, this, thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Stride run error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int StridedSliceCPUKernel::NormalRun() {
  auto input = in_tensors_.at(0);
  switch (input->data_type()) {
    case kNumberTypeInt8:
      param_->data_type = ::kNumberTypeInt8;
      break;
    case kNumberTypeFloat32:
      param_->data_type = ::kNumberTypeFloat32;
      break;
    case kNumberTypeFloat16:
      param_->data_type = ::kNumberTypeFloat16;
      break;
    case kNumberTypeInt32:
      param_->data_type = ::kNumberTypeInt32;
      break;
    default:
      MS_LOG(ERROR) << "Not supported data type: " << input->data_type();
      return RET_ERROR;
  }
  auto output = out_tensors_.at(0);
  CHECK_NULL_RETURN(input->data());
  CHECK_NULL_RETURN(output->data());
  auto ret = DoStridedSlice(input->data(), output->data(), param_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "StridedSlice error error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int StridedSliceCPUKernel::SoftCopyInputToOutput() {
  auto input_tensor = in_tensors().front();
  CHECK_NULL_RETURN(input_tensor);
  auto output_tensor = out_tensors().front();
  CHECK_NULL_RETURN(output_tensor);

  if (input_tensor->allocator() == nullptr || input_tensor->allocator() != output_tensor->allocator() ||
      input_tensor->allocator() != ms_context_->allocator || /* runtime allocator */
      op_parameter_->is_train_session_) {
    CHECK_NULL_RETURN(output_tensor->data());
    CHECK_NULL_RETURN(input_tensor->data());
    MS_CHECK_FALSE(input_tensor->Size() == 0, RET_ERROR);
    auto size = input_tensor->Size();
    thread_num_ = MSMIN(static_cast<size_t>(op_parameter_->thread_num_), UP_DIV(size, 16384));  // PerThreadMin : 16384
    if (thread_num_ < 1) {
      thread_num_ = 1;
    }
    auto block_size = UP_DIV(size, thread_num_);
    thread_num_ = UP_DIV(size, block_size);
    auto input_data = static_cast<const uint8_t *>(input_tensor->data());
    auto output_data = static_cast<uint8_t *>(output_tensor->data());
    auto Copy = [input_data, output_data, size, block_size, this](void *, int task_id, float, float) {
      auto in_start = input_data + task_id * block_size;
      auto out_start = output_data + task_id * block_size;
      auto copy_size = block_size;
      if (task_id == (thread_num_ - 1)) {
        copy_size = size - task_id * block_size;
      }
      (void)memcpy(out_start, in_start, copy_size);
      return RET_OK;
    };
    if (input_data != output_data) {
      if (thread_num_ == 1) {
        (void)memcpy(output_data, input_data, size);
        return RET_OK;
      }
      return lite::ParallelLaunch(this->ms_context_, Copy, nullptr, thread_num_);
    }
    return RET_OK;
  }

  output_tensor->FreeData();
  output_tensor->ResetRefCount();
  output_tensor->set_data(input_tensor->data());
  output_tensor->set_own_data(input_tensor->own_data());
  return RET_OK;
}

int StridedSliceCPUKernel::Run() {
  if (soft_copy_mode_) {
    return SoftCopyInputToOutput();
  }
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
