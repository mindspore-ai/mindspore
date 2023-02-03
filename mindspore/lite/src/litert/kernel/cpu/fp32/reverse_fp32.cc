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
#include "src/litert/kernel/cpu/fp32/reverse_fp32.h"
#include <cstring>
#include <vector>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/fp32/reverse_fp32.h"
#include "nnacl/errorcode.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ReverseV2;

namespace mindspore::kernel {
int ReverseCPUKernel::Stride(int index) {
  int stride = 1;
  for (size_t i = static_cast<size_t>(index) + 1; i < in_tensors_.at(0)->shape().size(); ++i) {
    stride *= in_tensors_.at(0)->shape().at(i);
  }
  return stride;
}

int ReverseCPUKernel::ReSize() {
  // trans negative to positive axis
  auto ret = UpdateAxisInfo();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get axis failed.";
    return ret;
  }

  data_size_ = in_tensors_.at(0)->ElementsNum();
  if (out_tensors_.at(0)->ElementsNum() != data_size_) {
    MS_LOG(ERROR) << "The number of outputs and inputs must be equal.";
    return RET_ERROR;
  }
  thread_sz_count_ = MSMIN(op_parameter_->thread_num_, data_size_);
  if (thread_sz_count_ == 0) {
    MS_LOG(ERROR) << "thread_sz_count_ can not be 0";
    return RET_ERROR;
  }
  thread_sz_stride_ = UP_DIV(data_size_, thread_sz_count_);

  auto *param = reinterpret_cast<ReverseParameter *>(op_parameter_);
  auto input_shape = in_tensors_.at(0)->shape();
  if (param->num_axis_ > static_cast<int>(input_shape.size())) {
    MS_LOG(ERROR) << "Reverse dims : " << param->num_axis_
                  << "is greater than input shape size :" << input_shape.size();
    return RET_ERROR;
  }
  if (input_shape.size() > REVERSE_SHAPE_MAX_SIZE) {
    MS_LOG(ERROR) << "input dimension num should <= " << REVERSE_SHAPE_MAX_SIZE;
    return RET_ERROR;
  }

  if (tmp_ != nullptr) {
    free(tmp_);
    tmp_ = nullptr;
  }
  MS_CHECK_INT_MUL_NOT_OVERFLOW(data_size_, static_cast<int>(sizeof(int)), RET_ERROR);
  tmp_ = reinterpret_cast<int *>(malloc(data_size_ * static_cast<int>(sizeof(int))));
  if (tmp_ == nullptr) {
    MS_LOG(ERROR) << "Reverse Malloc tmp_ error!";
    return RET_ERROR;
  }
  (void)memset(tmp_, 0, data_size_ * static_cast<int>(sizeof(int)));

  for (int i = 0; i < param->num_axis_; i++) {
    int axis = param->axis_[i];
    int stride = Stride(axis);
    strides_[i] = stride;
    inCount_[i] = input_shape[axis];
    outCount_[i] = 1;
    for (int j = 0; j < axis; j++) {
      outCount_[i] *= input_shape.at(j);
    }
  }

  int out;
  int in;
  int C;
  int m;
  for (int i = 0; i < data_size_; ++i) {
    int tmp = i;
    for (int j = 0; j < param->num_axis_; ++j) {
      C = inCount_[j];
      out = tmp / (C * strides_[j]);
      in = tmp / strides_[j] - out * C;
      m = tmp % strides_[j];
      tmp = out * C * strides_[j] + strides_[j] * (C - 1 - in) + m;
    }
    tmp_[i] = tmp;
  }

  return RET_OK;
}

int ReverseCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  auto reverse_param = reinterpret_cast<ReverseParameter *>(op_parameter_);
  CHECK_NULL_RETURN(reverse_param);
  CHECK_LESS_RETURN(reverse_param->num_axis_, 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  int ret = ReSize();
  return ret;
}

int ReverseRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto g_kernel = reinterpret_cast<ReverseCPUKernel *>(cdata);
  auto ret = g_kernel->DoReverse(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "reverseRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int ReverseCPUKernel::DoReverse(int task_id) {
  int count = MSMIN(thread_sz_stride_, data_size_ - task_id * thread_sz_stride_);
  if (count <= 0) {
    return RET_OK;
  }
  int offset = task_id * thread_sz_stride_;
  auto ret = Reverse(in_ptr_ + offset, out_ptr_, thread_sz_stride_, tmp_ + offset);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ReverseRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int ReverseCPUKernel::Run() {
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  in_ptr_ = reinterpret_cast<float *>(in_tensors_[0]->MutableData());
  CHECK_NULL_RETURN(in_ptr_);
  out_ptr_ = reinterpret_cast<float *>(out_tensors_[0]->MutableData());
  CHECK_NULL_RETURN(out_ptr_);
  auto ret = ParallelLaunch(this->ms_context_, ReverseRun, this, thread_sz_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Reverse run error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int ReverseCPUKernel::UpdateAxisInfo() {
  auto reverse_param = reinterpret_cast<ReverseParameter *>(op_parameter_);
  int in_shape_len = static_cast<int>(in_tensors_.front()->shape().size());
  for (int i = 0; i < reverse_param->num_axis_; ++i) {
    if (reverse_param->axis_[i] < 0) {
      reverse_param->axis_[i] += in_shape_len;
    }
    if (reverse_param->axis_[i] < 0 || reverse_param->axis_[i] >= in_shape_len) {
      MS_LOG(ERROR) << "Invalid axis.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ReverseV2, LiteKernelCreator<ReverseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_ReverseV2, LiteKernelCreator<ReverseCPUKernel>)
}  // namespace mindspore::kernel
