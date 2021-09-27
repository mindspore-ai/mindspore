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

#include "src/runtime/kernel/arm/fp32/log_softmax_fp32.h"
#include <cstring>
#include <vector>
#include "nnacl/fp32/log_softmax_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LogSoftmax;

namespace mindspore::kernel {
LogSoftmaxCPUKernel::~LogSoftmaxCPUKernel() {
  if (tmp_data_ != nullptr) {
    free(tmp_data_);
    tmp_data_ = nullptr;
  }
}

int LogSoftmaxCPUKernel::Init() {
  auto ret = SoftmaxBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int LogSoftmaxCPUKernel::ReSize() {
  auto ret = SoftmaxBaseCPUKernel::ReSize();
  if (ret != RET_OK) {
    return ret;
  }
  auto n_dim = softmax_param_->n_dim_;
  auto axis = softmax_param_->axis_;
  auto in_shape = in_tensors_.front()->shape();
  int out_plane_size = 1;
  for (int i = 0; i < axis; ++i) {
    out_plane_size *= in_shape.at(i);
  }
  int in_plane_size = 1;
  for (int i = axis + 1; i < n_dim; i++) {
    in_plane_size *= in_shape.at(i);
  }
  in_plane_size_ = in_plane_size;
  out_plane_size_ = out_plane_size;
  auto tmp_data_size =
    in_plane_size == 1 ? out_plane_size * in_plane_size * in_shape.at(axis) : out_plane_size * in_plane_size;
  if (tmp_data_ != nullptr) {
    free(tmp_data_);
  }
  tmp_data_ = reinterpret_cast<float *>(malloc(tmp_data_size * sizeof(float)));
  if (tmp_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc data for log_softmax fail!";
    return RET_ERROR;
  }
  return RET_OK;
}

int LogSoftmaxCPUKernel::DoLogSoftmaxLastAxis(int task_id) {
  MS_CHECK_FALSE(op_parameter_->thread_num_ == 0, RET_ERROR);
  int unit = UP_DIV(out_plane_size_, op_parameter_->thread_num_);
  int begin = task_id * unit;
  int end = MSMIN(begin + unit, out_plane_size_);
  int channel = softmax_param_->input_shape_[softmax_param_->axis_];
  int offset = begin * channel;
  auto input_ptr = reinterpret_cast<float *>(in_tensors_.at(kInputIndex)->data());
  CHECK_NULL_RETURN(input_ptr);
  auto output_ptr = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->data());
  CHECK_NULL_RETURN(output_ptr);
  LogSoftmaxLastAxis(input_ptr + offset, output_ptr + offset, tmp_data_ + offset, end - begin, channel);
  return RET_OK;
}

int LogSoftmaxLastAxisRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto kernel = reinterpret_cast<LogSoftmaxCPUKernel *>(cdata);
  CHECK_NULL_RETURN(kernel);
  auto ret = kernel->DoLogSoftmaxLastAxis(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoLogSoftmaxLastAxis error task_id: " << task_id << ", ret: " << ret;
  }
  return ret;
}

int LogSoftmaxCPUKernel::Run() {
  int ret = RET_OK;
  if (in_plane_size_ == 1) {
    ret = ParallelLaunch(this->ms_context_, LogSoftmaxLastAxisRun, this, op_parameter_->thread_num_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "LogSoftmaxCPUKernel ParallelLaunch failed, ret: " << ret;
    }
  } else {
    auto input_ptr = reinterpret_cast<float *>(in_tensors_.at(kInputIndex)->data());
    CHECK_NULL_RETURN(input_ptr);
    auto output_ptr = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->data());
    CHECK_NULL_RETURN(output_ptr);
    CHECK_NULL_RETURN(tmp_data_);
    LogSoftmax(input_ptr, output_ptr, tmp_data_, softmax_param_);
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LogSoftmax, LiteKernelCreator<LogSoftmaxCPUKernel>)
}  // namespace mindspore::kernel
