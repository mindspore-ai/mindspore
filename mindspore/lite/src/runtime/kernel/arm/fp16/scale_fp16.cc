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

#include "src/runtime/kernel/arm/fp16/scale_fp16.h"
#include <cstring>
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/kernel/arm/fp16/common_fp16.h"
#include "nnacl/fp16/scale_fp16.h"
#include "nnacl/fp16/cast_fp16.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ScaleFusion;

namespace mindspore::kernel {
int ScaleFp16CPUKernel::InitScaleOffset() {
  auto scale_tensor = in_tensors_.at(1);
  malloc_scale_ = scale_tensor->data_type() == kNumberTypeFloat32;

  if (in_tensors_.size() == 2) {
    malloc_offset_ = true;
  } else {
    auto offset_tensor = in_tensors_.at(2);
    malloc_offset_ = offset_tensor->data_type() == kNumberTypeFloat32;
  }
  return RET_OK;
}

int ScaleFp16CPUKernel::Init() {
  if (in_tensors_.size() < 2 || in_tensors_.size() > 3) {
    MS_LOG(ERROR) << "inputs to Scale operator should be 2 or 3, but " << in_tensors_.size() << " is given.";
    return RET_ERROR;
  }
  CHECK_LESS_RETURN(out_tensors_.size(), 1);

  if (!InferShapeDone()) {
    return RET_OK;
  }
  auto ret = ReSize();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale fp16 Resize failed";
    return RET_ERROR;
  }
  return RET_OK;
}

int ScaleFp16CPUKernel::ReSize() {
  auto ret = CalculateParameter();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale fp16 CalculateParameter failed.";
    return RET_ERROR;
  }

  return RET_OK;
}

int ScaleFp16CPUKernel::Scale(int task_id) {
  switch (scale_param_->activation_type_) {
    case schema::ActivationType_RELU6:
      DoScaleRelu6Fp16(input_, output_, scale_, offset_, task_id, scale_param_);
      break;
    case schema::ActivationType_RELU:
      Fp16DoScaleRelu(input_, output_, scale_, offset_, task_id, scale_param_);
      break;
    case schema::ActivationType_NO_ACTIVATION:
      DoScaleFp16(input_, output_, scale_, offset_, task_id, scale_param_);
      break;
    default:
      MS_LOG(ERROR) << "ScaleFp16 does not support activation type " << scale_param_->activation_type_;
      return RET_ERROR;
  }
  return RET_OK;
}

int ScaleFp16Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto scale = reinterpret_cast<ScaleFp16CPUKernel *>(cdata);
  auto ret = scale->Scale(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ScaleRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ScaleFp16CPUKernel::Run() {
  auto input_tensor = in_tensors_.at(0);
  auto output_tensor = out_tensors_.at(0);
  CHECK_NULL_RETURN(input_tensor);
  CHECK_NULL_RETURN(output_tensor);
  input_ = reinterpret_cast<float16_t *>(input_tensor->data());
  output_ = reinterpret_cast<float16_t *>(output_tensor->data());
  CHECK_NULL_RETURN(input_);
  CHECK_NULL_RETURN(output_);
  auto ret = InitScaleOffset();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale fp16 InitScaleOffset failed.";
    return RET_ERROR;
  }

  ret = MallocAssignTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale Fp16 malloc tmp buffer failed";
    FreeTmpBuffer();
    return ret;
  }

  ret = ParallelLaunch(this->ms_context_, ScaleFp16Run, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale error error_code[" << ret << "]";
    FreeTmpBuffer();
    return RET_ERROR;
  }

  FreeTmpBuffer();
  return RET_OK;
}

int ScaleFp16CPUKernel::MallocAssignTmpBuffer() {
  scale_ = ConvertInputFp32toFp16(in_tensors_.at(1), static_cast<const lite::InnerContext *>(this->ms_context_));
  if (scale_ == nullptr) {
    return RET_ERROR;
  }
  if (in_tensors_.size() == 3) {
    offset_ = ConvertInputFp32toFp16(in_tensors_.at(2), static_cast<const lite::InnerContext *>(this->ms_context_));
    if (offset_ == nullptr) {
      return RET_ERROR;
    }
  } else {
    MS_CHECK_INT_MUL_NOT_OVERFLOW(in_tensors_.at(1)->ElementsNum(), sizeof(float16_t), RET_ERROR);
    offset_ = reinterpret_cast<float16_t *>(
      ms_context_->allocator->Malloc(in_tensors_.at(1)->ElementsNum() * sizeof(float16_t)));
    if (offset_ == nullptr) {
      MS_LOG(ERROR) << "Malloc data failed";
      return RET_ERROR;
    }
    memset(offset_, 0, in_tensors_.at(1)->ElementsNum() * sizeof(float16_t));
  }
  return RET_OK;
}

void ScaleFp16CPUKernel::FreeTmpBuffer() {
  if (malloc_scale_ && scale_ != nullptr) {
    ms_context_->allocator->Free(scale_);
    scale_ = nullptr;
  }
  if (malloc_offset_ && offset_ != nullptr) {
    ms_context_->allocator->Free(offset_);
    offset_ = nullptr;
  }
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_ScaleFusion, LiteKernelCreator<ScaleFp16CPUKernel>)
}  // namespace mindspore::kernel
