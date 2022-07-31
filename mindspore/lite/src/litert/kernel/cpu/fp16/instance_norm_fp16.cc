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
#include "src/litert/kernel/cpu/fp16/instance_norm_fp16.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp16/cast_fp16.h"
#include "nnacl/fp16/instance_norm_fp16.h"
#include "nnacl/fp16/pack_fp16.h"

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

int InstanceNormFp16CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C3NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
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
  if (input_pack_to_nc8hw8_) {
    ret = InstanceNormNC8HW8Fp16(tmp_src_data_, dst_data_, gamma_data_, beta_data_, param_, task_id);
  } else {
    ret = InstanceNormFp16(tmp_src_data_, dst_data_, gamma_data_, beta_data_, param_, task_id);
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

  CHECK_NULL_RETURN(in_tensors_.at(1));
  gamma_data_ = reinterpret_cast<float16_t *>(in_tensors_.at(1)->data());
  CHECK_NULL_RETURN(gamma_data_);
  CHECK_NULL_RETURN(in_tensors_.at(2));
  beta_data_ = reinterpret_cast<float16_t *>(in_tensors_.at(2)->data());
  CHECK_NULL_RETURN(beta_data_);

  if (in_tensors_[0]->format() == NHWC) {
    tmp_src_data_ = reinterpret_cast<float16_t *>(ms_context_->allocator->Malloc(in_tensors_[0]->Size()));
    CHECK_NULL_RETURN(tmp_src_data_);
    PackNHWCToNC8HW8NotAlignedFp16(src_data_, tmp_src_data_, param_->batch_, param_->inner_size_, param_->channel_);
    input_pack_to_nc8hw8_ = true;
  } else if (in_tensors_[0]->format() == NC4HW4) {
    input_pack_to_nc8hw8_ = true;
    tmp_src_data_ = src_data_;
  } else {
    tmp_src_data_ = src_data_;
  }
  auto ret = ParallelLaunch(this->ms_context_, InstanceNormFp16Run, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InstanceNormFp16Run error error_code[" << ret << "]";
    return ret;
  }
  if (tmp_src_data_ != src_data_) {
    FreeTmpSrcData();
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_InstanceNorm, LiteKernelCreator<InstanceNormFp16CPUKernel>)
}  // namespace mindspore::kernel
