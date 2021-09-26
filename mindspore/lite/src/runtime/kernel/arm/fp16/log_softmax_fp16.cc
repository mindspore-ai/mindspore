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

#include "src/runtime/kernel/arm/fp16/log_softmax_fp16.h"
#include <cstring>
#include <vector>
#include "src/runtime/kernel/arm/fp16/common_fp16.h"
#include "nnacl/fp16/log_softmax_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LogSoftmax;

namespace mindspore::kernel {
LogSoftmaxFp16CPUKernel::~LogSoftmaxFp16CPUKernel() {
  if (tmp_data_ != nullptr) {
    free(tmp_data_);
    tmp_data_ = nullptr;
  }
}

int LogSoftmaxFp16CPUKernel::Init() {
  auto ret = SoftmaxBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int LogSoftmaxFp16CPUKernel::ReSize() {
  auto ret = SoftmaxBaseCPUKernel::ReSize();
  if (ret != RET_OK) {
    return ret;
  }
  auto n_dim = softmax_param_->n_dim_;
  auto axis = softmax_param_->axis_;
  auto in_shape = in_tensors_.front()->shape();
  out_plane_size_ = 1;
  for (int i = 0; i < axis; ++i) {
    out_plane_size_ *= in_shape[i];
  }
  in_plane_size_ = 1;
  for (int i = axis + 1; i < n_dim; i++) {
    in_plane_size_ *= in_shape[i];
  }
  auto tmp_data_size =
    in_plane_size_ == 1 ? out_plane_size_ * in_plane_size_ * in_shape.at(axis) : out_plane_size_ * in_plane_size_;
  if (tmp_data_ != nullptr) {
    free(tmp_data_);
  }
  tmp_data_ = reinterpret_cast<float16_t *>(malloc(tmp_data_size * sizeof(float16_t)));
  if (tmp_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc data for softmax fail!";
    return RET_ERROR;
  }
  return RET_OK;
}

int LogSoftmaxFp16CPUKernel::DoLogSoftmaxLastAxis(int task_id) {
  MS_CHECK_FALSE(op_parameter_->thread_num_ == 0, RET_ERROR);
  int unit = UP_DIV(out_plane_size_, op_parameter_->thread_num_);
  int begin = task_id * unit;
  int end = MSMIN(begin + unit, out_plane_size_);
  int channel = softmax_param_->input_shape_[softmax_param_->axis_];
  int offset = begin * channel;
  auto input_ptr = reinterpret_cast<float16_t *>(in_tensors_.at(kInputIndex)->data());
  CHECK_NULL_RETURN(input_ptr);
  auto output_ptr = reinterpret_cast<float16_t *>(out_tensors_.at(kOutputIndex)->data());
  CHECK_NULL_RETURN(output_ptr);
  LogSoftmaxLastAxisFp16(input_ptr + offset, output_ptr + offset, tmp_data_ + offset, end - begin, channel);
  return RET_OK;
}

int LogSoftmaxLastAxisFp16Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto kernel = reinterpret_cast<LogSoftmaxFp16CPUKernel *>(cdata);
  CHECK_NULL_RETURN(kernel);
  auto ret = kernel->DoLogSoftmaxLastAxis(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoLogSoftmaxLastAxisFp16 error task_id: " << task_id << ", ret: " << ret;
  }
  return ret;
}

int LogSoftmaxFp16CPUKernel::Run() {
  if (in_plane_size_ == 1) {
    auto ret = ParallelLaunch(this->ms_context_, LogSoftmaxLastAxisFp16Run, this, op_parameter_->thread_num_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "LogSoftmaxFp16CPUKernel ParallelLaunch failed, ret: " << ret;
    }
    return ret;
  } else {
    input_fp16_ = reinterpret_cast<float16_t *>(in_tensors_.at(0)->data());
    CHECK_NULL_RETURN(input_fp16_);
    output_fp16_ = reinterpret_cast<float16_t *>(out_tensors_.at(0)->data());
    CHECK_NULL_RETURN(output_fp16_);
    CHECK_NULL_RETURN(tmp_data_);
    LogSoftmaxFp16(input_fp16_, output_fp16_, tmp_data_, softmax_param_);
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_LogSoftmax, LiteKernelCreator<LogSoftmaxFp16CPUKernel>)
}  // namespace mindspore::kernel
