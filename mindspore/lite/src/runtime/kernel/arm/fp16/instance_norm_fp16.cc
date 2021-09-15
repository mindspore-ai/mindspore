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
#include "src/runtime/kernel/arm/fp16/instance_norm_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp16/cast_fp16.h"
#include "nnacl/fp16/instance_norm_fp16.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_InstanceNorm;

namespace mindspore::kernel {
void InstanceNormFp16CPUKernel::FreeTmpBuffer() {
  if (in_tensors_[1]->data_type() == kNumberTypeFloat32) {
    if (gamma_data_ != nullptr) {
      free(gamma_data_);
      gamma_data_ = nullptr;
    }
  }
  if (in_tensors_[2]->data_type() == kNumberTypeFloat32) {
    if (beta_data_ != nullptr) {
      free(beta_data_);
      beta_data_ = nullptr;
    }
  }
}

int InstanceNormFp16CPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 3);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  auto gamma = in_tensors_.at(1);
  CHECK_NULL_RETURN(gamma->data());
  if (gamma->data_type() == kNumberTypeFloat32) {
    gamma_data_ = reinterpret_cast<float16_t *>(malloc(gamma->ElementsNum() * sizeof(float16_t)));
    if (gamma_data_ == nullptr) {
      MS_LOG(ERROR) << "InstanceNorm fp16 kernel malloc gamma_data_ error.";
      return RET_ERROR;
    }
    Float32ToFloat16(reinterpret_cast<float *>(gamma->data()), gamma_data_, gamma->ElementsNum());
  } else if (gamma->data_type() == kNumberTypeFloat16) {
    gamma_data_ = reinterpret_cast<float16_t *>(gamma->data());
  } else {
    MS_LOG(ERROR) << "Unsupported data type of gamma tensor for instance norm.";
    return RET_ERROR;
  }

  auto beta = in_tensors_.at(2);
  CHECK_NULL_RETURN(beta->data());
  if (beta->data_type() == kNumberTypeFloat32) {
    beta_data_ = reinterpret_cast<float16_t *>(malloc(beta->ElementsNum() * sizeof(float16_t)));
    if (beta_data_ == nullptr) {
      MS_LOG(ERROR) << "InstanceNorm fp16 kernel malloc beta_data_ error.";
      return RET_ERROR;
    }
    Float32ToFloat16(reinterpret_cast<float *>(beta->data()), beta_data_, beta->ElementsNum());
  } else if (beta->data_type() == kNumberTypeFloat16) {
    beta_data_ = reinterpret_cast<float16_t *>(beta->data());
  } else {
    MS_LOG(ERROR) << "Unsupported data type of beta tensor for instance norm.";
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int InstanceNormFp16CPUKernel::ReSize() {
  param_->op_parameter_.thread_num_ = op_parameter_->thread_num_;
  auto in_tensor = in_tensors_.front();
  param_->batch_ = in_tensor->Batch();
  param_->inner_size_ = in_tensor->Height() * in_tensor->Width();
  param_->channel_ = in_tensor->Channel();
  return RET_OK;
}

int InstanceNormFp16CPUKernel::DoInstanceNorm(int task_id) {
  int ret = RET_OK;
  if (in_tensors_.at(0)->format() == NC4HW4) {
    ret = InstanceNormNC8HW8Fp16(src_data_, dst_data_, gamma_data_, beta_data_, param_, task_id);
  } else {
    ret = InstanceNormFp16(src_data_, dst_data_, gamma_data_, beta_data_, param_, task_id);
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoInstanceNorm error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int InstanceNormFp16Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto kernel = reinterpret_cast<InstanceNormFp16CPUKernel *>(cdata);
  auto ret = kernel->DoInstanceNorm(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InstanceNormFp16Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int InstanceNormFp16CPUKernel::Run() {
  src_data_ = reinterpret_cast<float16_t *>(in_tensors_.at(0)->data());
  dst_data_ = reinterpret_cast<float16_t *>(out_tensors_.at(0)->data());
  CHECK_NULL_RETURN(src_data_);
  CHECK_NULL_RETURN(dst_data_);
  auto ret = ParallelLaunch(this->ms_context_, InstanceNormFp16Run, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InstanceNormFp16Run error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_InstanceNorm, LiteKernelCreator<InstanceNormFp16CPUKernel>)
}  // namespace mindspore::kernel
