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
int ScaleCPUKernel::InitScaleOffset() {
  auto param = reinterpret_cast<ScaleParameter *>(opParameter);
  auto scale_tensor = inputs_.at(1);
  float *scale_ptr = reinterpret_cast<float *>(inputs_.at(1)->Data());
  if (scale_ptr != nullptr) {
    scale_ = reinterpret_cast<float *>(malloc(scale_tensor->ElementsNum() * sizeof(float)));
    if (scale_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
    memcpy(scale_, scale_ptr, scale_tensor->ElementsNum() * sizeof(float));
  } else {
    scale_ = nullptr;
  }

  if (inputs_.size() == 3) {
    auto offset_tensor = inputs_.at(1);
    offset_ = reinterpret_cast<float *>(malloc(offset_tensor->ElementsNum() * sizeof(float)));
    if (offset_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
    param->has_offset_ = true;
  } else {
    offset_ = nullptr;
    param->has_offset_ = false;
  }
  return RET_OK;
}

int ScaleCPUKernel::InitParameter() {
  auto param = reinterpret_cast<ScaleParameter *>(opParameter);
  auto in_tensor = inputs_.at(0);
  auto in_shape = in_tensor->shape();
  auto scale_tensor = inputs_.at(1);
  auto scale_shape = scale_tensor->shape();

  if (scale_shape.size() + param->axis_ > in_shape.size()) {
    MS_LOG(ERROR) << "Scale tensor shape is incorrect.";
    return RET_ERROR;
  }
  param->outer_size_ = 1;
  param->axis_size_ = 1;
  param->inner_size_ = 1;
  for (int i = 0; i < param->axis_; i++) {
    param->outer_size_ *= in_shape[i];
  }
  for (int i = 0; i < scale_shape.size(); i++) {
    if (in_shape[i + param->axis_] != scale_shape[i]) {
      MS_LOG(ERROR) << "Scale tensor shape is incorrect.";
      return RET_ERROR;
    }
    param->axis_size_ *= in_shape[i + param->axis_];
  }
  for (int i = param->axis_ + scale_shape.size(); i < in_shape.size(); i++) {
    param->inner_size_ *= in_shape[i];
  }
  return RET_OK;
}

int ScaleCPUKernel::Init() {
  if (inputs_.size() < 2 || inputs_.size() > 3) {
    MS_LOG(ERROR) << "inputs to Scale operator should be 2 or 3, but " << inputs_.size() << " is given.";
    return RET_ERROR;
  }

  auto ret = InitParameter();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale fp32 InitParameter failed.";
    return RET_ERROR;
  }

  ret = InitScaleOffset();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale fp32 InitScaleOffset failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ScaleCPUKernel::ReSize() { return RET_OK; }

int ScaleCPUKernel::Scale(int task_id) {
  auto ret =
    DoScale(input_ptr_, output_ptr_, scale_, offset_, task_id, reinterpret_cast<ScaleParameter *>(opParameter));

  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ScaleRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto scale = reinterpret_cast<ScaleCPUKernel *>(cdata);
  auto ret = scale->Scale(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ScaleRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ScaleCPUKernel::Run() {
  auto in_tensor = inputs_.front();
  input_ptr_ = reinterpret_cast<float *>(in_tensor->Data());
  if (scale_ == nullptr) {
    auto scale_tensor = inputs_[1];
    scale_ = reinterpret_cast<float *>(scale_tensor->Data());
  }
  auto out_tensor = outputs_.front();
  output_ptr_ = reinterpret_cast<float *>(out_tensor->Data());

  int ret = LiteBackendParallelLaunch(ScaleRun, this, opParameter->thread_num_);
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Scale, CpuScaleFp32KernelCreator)
}  // namespace mindspore::kernel
