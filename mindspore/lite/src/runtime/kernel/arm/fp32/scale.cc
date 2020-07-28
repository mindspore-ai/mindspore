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

#include "src/runtime/kernel/arm/fp32/scale.h"
#include <string.h>
#include <vector>
#include "src/runtime/kernel/arm/opclib/scale.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Scale;

namespace mindspore::kernel {
namespace {
constexpr int kScaleInputNum = 1;
constexpr int kScaleOutputNum = 1;
}  // namespace
int ScaleCPUKernel::Init() {
  auto param = reinterpret_cast<ScaleParameter *>(opParameter);
  auto in_tensor = inputs_.front();
  auto scale = inputs_.at(1);

  if (inputs_.size() < 2 || inputs_.size() > 3) {
    MS_LOG(ERROR) << "inputs to Scale operator should be 2 or 3, but " << inputs_.size() << " is given.";
    return RET_ERROR;
  }

  if (param->axis_ < 0) {
    MS_LOG(ERROR) << "axis illegal.";
    return RET_ERROR;
  }
  if (param->num_axis_ < 1 || param->num_axis_ + param->axis_ >= in_tensor->shape().size()) {
    MS_LOG(ERROR) << "number of axis illegal";
    return RET_ERROR;
  }

  param->channel_ = 1;
  param->out_count_ = 1;
  param->in_stride_ = 1;
  int cur_axis;
  for (cur_axis = 0; cur_axis < param->axis_; cur_axis++) {
    param->out_count_ *= in_tensor->shape()[cur_axis];
  }
  for (int i = 0; i < param->num_axis_; i++) {
    param->channel_ *= in_tensor->shape()[(cur_axis++)];
  }
  for (int i = cur_axis; i < in_tensor->shape().size(); i++) {
    param->in_stride_ *= in_tensor->shape()[cur_axis];
  }
  if (scale->shape().back() != param->channel_ || scale->shape().size() > 2) {
    MS_LOG(ERROR) << "scale shape illegal.";
    return RET_ERROR;
  }
  if (inputs_.size() == 3) {
    if ((inputs_.at(2))->shape().back() != param->channel_ || (inputs_.at(2))->shape().size() > 2) {
      MS_LOG(ERROR) << "offset shape illegal.";
      return RET_ERROR;
    }
  }

  input_ptr_ = reinterpret_cast<float *>(inputs_.front()->Data());
  scale_ = reinterpret_cast<float *>(inputs_.at(1)->Data());
  if (inputs_.size() == 3) {
    offset_ = reinterpret_cast<float *>(inputs_.at(2)->Data());
    has_offset_ = true;
  } else {
    offset_ = nullptr;
    has_offset_ = false;
  }
  output_ptr_ = reinterpret_cast<float *>(outputs_.front()->Data());

  num_unit_ = param->out_count_ * param->channel_;
  unit_size_ = param->in_stride_;
  thread_n_num_ = MSMIN(thread_num_, num_unit_);
  thread_n_stride_ = UP_DIV(num_unit_, thread_n_num_);
  return RET_OK;
}

int ScaleCPUKernel::Scale(int task_id) {
  int num_unit_thread = MSMIN(thread_n_stride_, num_unit_ - task_id * thread_n_stride_);
  if (num_unit_thread <= 0) {
    return RET_OK;
  }
  int thread_offset = task_id * thread_n_stride_;
  int ret;
  if (has_offset_) {
    ret = DoScale(input_ptr_, output_ptr_, scale_, offset_, thread_offset, num_unit_thread,
                  reinterpret_cast<ScaleParameter *>(opParameter));
  } else {
    ret = DoScale(input_ptr_, output_ptr_, scale_, thread_offset, num_unit_thread,
                  reinterpret_cast<ScaleParameter *>(opParameter));
  }

  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ScaleCPUKernel::ReSize() { return RET_OK; }

int ScaleRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto g_kernel = reinterpret_cast<ScaleCPUKernel *>(cdata);
  auto ret = g_kernel->Scale(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ScaleRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ScaleCPUKernel::Run() {
  int ret = LiteBackendParallelLaunch(ScaleRun, this, thread_n_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale error error_code[" << ret << "]";
    return RET_ERROR;
  }

  return RET_OK;
}

kernel::LiteKernel *CpuScaleFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                              const std::vector<lite::tensor::Tensor *> &outputs,
                                              OpParameter *opParameter, const lite::Context *ctx,
                                              const kernel::KernelKey &desc) {
  MS_ASSERT(desc.type == schema::PrimitiveType_Scale);
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "opParameter is nullptr";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) ScaleCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "New kernel fails.";
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }

  return kernel;
}

REG_KERNEL(kCPU, PrimitiveType_Scale, CpuScaleFp32KernelCreator)
}  // namespace mindspore::kernel
