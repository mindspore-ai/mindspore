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
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Scale;

namespace mindspore::kernel {
ScaleCPUKernel::~ScaleCPUKernel() {
  if (scale_param_->const_scale_) {
    if (scale_ != nullptr) {
      free(scale_);
      scale_ = nullptr;
    }
  }
  if (scale_param_->has_offset_) {
    if (offset_ != nullptr) {
      free(offset_);
      offset_ = nullptr;
    }
  }
}

int ScaleCPUKernel::InitScaleOffset() {
  auto scale_tensor = in_tensors_.at(1);
  float *scale_ptr = reinterpret_cast<float *>(in_tensors_.at(1)->Data());
  if (scale_ptr != nullptr) {
    scale_param_->const_scale_ = true;
    scale_ = reinterpret_cast<float *>(malloc(scale_tensor->ElementsNum() * sizeof(float)));
    if (scale_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
    memcpy(scale_, scale_ptr, scale_tensor->ElementsNum() * sizeof(float));
  } else {
    scale_param_->const_scale_ = false;
    scale_ = nullptr;
  }

  if (in_tensors_.size() == 3) {
    auto offset_tensor = in_tensors_.at(2);
    offset_ = reinterpret_cast<float *>(malloc(offset_tensor->ElementsNum() * sizeof(float)));
    if (offset_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
    memcpy(offset_, offset_tensor->Data(), offset_tensor->ElementsNum() * sizeof(float));
    scale_param_->has_offset_ = true;
  } else {
    offset_ = nullptr;
    scale_param_->has_offset_ = false;
  }
  return RET_OK;
}

int ScaleCPUKernel::InitParameter() {
  auto in_tensor = in_tensors_.at(0);
  auto in_shape = in_tensor->shape();
  auto scale_tensor = in_tensors_.at(1);
  auto scale_shape = scale_tensor->shape();

  if (scale_shape.size() + scale_param_->axis_ > in_shape.size()) {
    MS_LOG(ERROR) << "Scale tensor shape is incorrect.";
    return RET_ERROR;
  }
  scale_param_->outer_size_ = 1;
  scale_param_->axis_size_ = 1;
  scale_param_->inner_size_ = 1;
  for (int i = 0; i < scale_param_->axis_; i++) {
    scale_param_->outer_size_ *= in_shape[i];
  }
  for (int i = 0; i < scale_shape.size(); i++) {
    if (in_shape[i + scale_param_->axis_] != scale_shape[i]) {
      MS_LOG(ERROR) << "Scale tensor shape is incorrect.";
      return RET_ERROR;
    }
    scale_param_->axis_size_ *= in_shape[i + scale_param_->axis_];
  }
  for (int i = scale_param_->axis_ + scale_shape.size(); i < in_shape.size(); i++) {
    scale_param_->inner_size_ *= in_shape[i];
  }
  return RET_OK;
}

int ScaleCPUKernel::Init() {
  if (in_tensors_.size() < 2 || in_tensors_.size() > 3) {
    MS_LOG(ERROR) << "inputs to Scale operator should be 2 or 3, but " << in_tensors_.size() << " is given.";
    return RET_ERROR;
  }

  if (!InferShapeDone()) {
    return RET_OK;
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

int ScaleCPUKernel::ReSize() {
  auto ret = InitParameter();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale fp32 InitParameter failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ScaleCPUKernel::Scale(int task_id) {
  auto ret = DoScale(input_ptr_, output_ptr_, scale_, offset_, task_id, scale_param_);
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
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return ret;
  }
  auto in_tensor = in_tensors_.front();
  input_ptr_ = reinterpret_cast<float *>(in_tensor->Data());
  if (scale_ == nullptr) {
    auto scale_tensor = in_tensors_[1];
    scale_ = reinterpret_cast<float *>(scale_tensor->Data());
  }
  auto out_tensor = out_tensors_.front();
  output_ptr_ = reinterpret_cast<float *>(out_tensor->Data());

  ret = LiteBackendParallelLaunch(ScaleRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale error error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuScaleFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                              const std::vector<lite::tensor::Tensor *> &outputs,
                                              OpParameter *opParameter, const lite::Context *ctx,
                                              const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(desc.type == schema::PrimitiveType_Scale);
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "opParameter is nullptr";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) ScaleCPUKernel(opParameter, inputs, outputs, ctx, primitive);
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
