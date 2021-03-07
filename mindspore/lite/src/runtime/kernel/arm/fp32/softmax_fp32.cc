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

#include "src/runtime/kernel/arm/fp32/softmax_fp32.h"
#include <string.h>
#include <vector>
#include "nnacl/fp32/softmax_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Softmax;

namespace mindspore::kernel {
int SoftmaxCPUKernel::Init() {
  auto ret = SoftmaxBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int SoftmaxCPUKernel::ReSize() {
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
  if (sum_data_ != nullptr) {
    free(sum_data_);
  }
  sum_data_ = reinterpret_cast<float *>(malloc(out_plane_size * in_plane_size * sizeof(float)));
  if (sum_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc data for softmax fail!";
    return RET_ERROR;
  }
  return RET_OK;
}

int SoftmaxCPUKernel::DoSoftmaxLastAxis(int task_id) {
  int unit = UP_DIV(out_plane_size_, context_->thread_num_);
  int begin = task_id * unit;
  int end = MSMIN(begin + unit, out_plane_size_);
  int channel = softmax_param_->input_shape_[softmax_param_->axis_];
  int offset = begin * channel;
  auto input_ptr = reinterpret_cast<float *>(in_tensors_.at(kInputIndex)->MutableData());
  auto output_ptr = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->MutableData());
  SoftmaxLastAxis(input_ptr + offset, output_ptr + offset, end - begin, channel);
  return RET_OK;
}

int SoftmaxLastAxisRun(void *cdata, int task_id) {
  auto kernel = reinterpret_cast<SoftmaxCPUKernel *>(cdata);
  auto ret = kernel->DoSoftmaxLastAxis(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoSoftmaxLastAxis error task_id: " << task_id << ", ret: " << ret;
  }
  return ret;
}

int SoftmaxCPUKernel::Run() {
  auto input_ptr = reinterpret_cast<float *>(in_tensors_.at(kInputIndex)->MutableData());
  MS_ASSERT(input_ptr);
  auto output_ptr = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->MutableData());
  MS_ASSERT(output_ptr);
  int ret = RET_OK;
  if (in_plane_size_ == 1) {
    ret = ParallelLaunch(this->context_->thread_pool_, SoftmaxLastAxisRun, this, context_->thread_num_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "SoftmaxCPUKernel ParallelLaunch failed, ret: " << ret;
    }
  } else {
    MS_ASSERT(sum_data_);
    memset(sum_data_, 0, in_plane_size_ * out_plane_size_ * sizeof(float));
    MS_ASSERT(softmax_param_);
    Softmax(input_ptr, output_ptr, sum_data_, softmax_param_);
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Softmax, LiteKernelCreator<SoftmaxCPUKernel>)
}  // namespace mindspore::kernel
