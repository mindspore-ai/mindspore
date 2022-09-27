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
#include "src/litert/kernel/cpu/base/split_base.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Split;

namespace mindspore::kernel {
int SplitBaseCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(param);
  output_ptr_.resize(param->num_split_);
  for (size_t i = 0; i < output_ptr_.size(); i++) {
    output_ptr_.at(i) = nullptr;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int SplitBaseCPUKernel::CheckAndInitSplitParam(const lite::Tensor &in_tensor, SplitParameter *param) {
  auto input_shape = in_tensor.shape();
  CHECK_LESS_RETURN(input_shape.size(), 1);
  CHECK_LESS_RETURN(SPLIT_STRIDES_SIZE - 1, input_shape.size());
  auto split_dim = param->split_dim_;
  param->split_dim_ = split_dim >= 0 ? split_dim : static_cast<int>(input_shape.size()) + split_dim;
  param->strides_[input_shape.size() - 1] = 1;
  for (int i = static_cast<int>(input_shape.size()) - 2; i >= 0; i--) {
    MS_CHECK_FALSE(INT_MUL_OVERFLOW(param->strides_[i + 1], input_shape.at(i + 1)), RET_ERROR);
    param->strides_[i] = param->strides_[i + 1] * input_shape.at(i + 1);
  }
  CHECK_LESS_RETURN(static_cast<int>(input_shape.size()), param->split_dim_ + 1);
  if (input_shape.at(param->split_dim_) == 0) {
    MS_LOG(ERROR) << "input_shape[" << param->split_dim_ << "] must not be zero!";
    return RET_ERROR;
  }
  CHECK_LESS_RETURN(SPLIT_STRIDES_SIZE, param->split_dim_ + 1);
  if (param->strides_[param->split_dim_] == 0) {
    MS_LOG(ERROR) << "param->strides_[" << param->split_dim_ << "] must not be zero!";
    return RET_ERROR;
  }
  CHECK_LESS_RETURN((input_shape.at(param->split_dim_) * param->strides_[param->split_dim_]), 1);

  MS_CHECK_FALSE(INT_MUL_OVERFLOW(param->strides_[0], input_shape.at(0)), RET_ERROR);
  param->split_count_ =
    param->strides_[0] * input_shape.at(0) / (input_shape.at(param->split_dim_) * param->strides_[param->split_dim_]);
  param->n_dims_ = static_cast<int>(input_shape.size());
  CHECK_LESS_RETURN(param->num_split_, 1);
  CHECK_LESS_RETURN(input_shape[param->split_dim_], static_cast<int>(param->num_split_));
  if (param->split_sizes_[0] == 0) {
    if (input_shape[param->split_dim_] % param->num_split_ != 0) {
      MS_LOG(ERROR) << "Default split size is not usable.";
      return RET_ERROR;
    }
    int split_size = input_shape.at(param->split_dim_) / param->num_split_;
    for (int i = 0; i < param->num_split_; i++) {
      param->split_sizes_[i] = split_size;
    }
  } else {
    int64_t split_sizes_sum = 0;
    for (int i = 0; i < param->num_split_; i++) {
      split_sizes_sum += param->split_sizes_[i];
    }
    if (split_sizes_sum > input_shape[param->split_dim_]) {
      MS_LOG(ERROR) << "Customer-based split sizes is not usable.";
      return RET_ERROR;
    }
  }

  if (param->split_sizes_[param->num_split_ - 1] == -1) {
    int split_shape_end = input_shape.at(param->split_dim_);
    for (int i = 0; i < param->num_split_ - 1; i++) {
      split_shape_end -= param->split_sizes_[i];
    }
    param->split_sizes_[param->num_split_ - 1] = split_shape_end;
  }
  return RET_OK;
}

int SplitBaseCPUKernel::ReSize() {
  auto in_tensor = in_tensors_.front();
  CHECK_NULL_RETURN(in_tensor);
  auto status = CheckAndInitSplitParam(*in_tensor, param);
  if (RET_OK != status) {
    MS_LOG(ERROR) << "CheckAndInitSplitParam failed";
    return status;
  }

  // split_count means the previous dims num before split dim
  // e.g. input dims is [1, 3, 4, 8], split axis is 2, num_split is 2, so split_count_ is 1*3, num_unit_ is 1*3*2
  MS_CHECK_FALSE(INT_MUL_OVERFLOW(param->split_count_, param->num_split_), RET_ERROR);
  num_unit_ = param->split_count_ * param->num_split_;

  if (UpdateThreadNumPass(TC_PTYPE(type_), 1, 1, out_tensors_.at(0)->ElementsNum()) != RET_OK) {
    return RET_ERROR;
  }

  thread_num_ = MSMIN(thread_num_, num_unit_);
  if (thread_num_ != 0) {
    thread_n_stride_ = UP_DIV(num_unit_, thread_num_);
  }
  return RET_OK;
}

int SplitBaseCPUKernel::Split(int task_id) {
  MS_CHECK_FALSE(INT_MUL_OVERFLOW(task_id, thread_n_stride_), RET_ERROR);
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

static int SplitRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto g_kernel = reinterpret_cast<SplitBaseCPUKernel *>(cdata);
  CHECK_NULL_RETURN(g_kernel);
  auto ret = g_kernel->Split(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SplitRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SplitBaseCPUKernel::Run() {
  auto input_tensor = in_tensors_.at(0);
  input_ptr_ = input_tensor->data();
  if (input_ptr_ == nullptr) {
    return RET_NULL_PTR;
  }

  for (int i = 0; i < param->num_split_; i++) {
    auto output_tensor = out_tensors_.at(i);
    output_ptr_.at(i) = output_tensor->data();
    if (output_ptr_.at(i) == nullptr) {
      return RET_NULL_PTR;
    }
  }

  auto ret = ParallelLaunch(this->ms_context_, SplitRun, this, thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "split error error_code[" << ret << "]";
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Split, LiteKernelCreator<SplitBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Split, LiteKernelCreator<SplitBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Split, LiteKernelCreator<SplitBaseCPUKernel>)
}  // namespace mindspore::kernel
