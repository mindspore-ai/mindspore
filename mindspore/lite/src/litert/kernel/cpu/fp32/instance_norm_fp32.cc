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
#include "src/litert/kernel/cpu/fp32/instance_norm_fp32.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp32/instance_norm_fp32.h"
#include "nnacl/fp32/pack_fp32.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_InstanceNorm;

namespace mindspore::kernel {
int InstanceNormCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_3D);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);

  const auto &input_tensor = in_tensors_[FIRST_INPUT];
  const auto &gamma_tensor = in_tensors_[SECOND_INPUT];
  const auto &beta_tensor = in_tensors_[THIRD_INPUT];
  CHECK_NOT_EQUAL_RETURN(input_tensor->data_type(), kNumberTypeFloat32);
  CHECK_NOT_EQUAL_RETURN(gamma_tensor->data_type(), kNumberTypeFloat32);
  CHECK_NOT_EQUAL_RETURN(beta_tensor->data_type(), kNumberTypeFloat32);

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int InstanceNormCPUKernel::ReSize() {
  const auto &in_tensor = in_tensors_[FIRST_INPUT];
  const auto &gamma_tensor = in_tensors_[SECOND_INPUT];
  const auto &beta_tensor = in_tensors_[THIRD_INPUT];
  auto channel = in_tensor->Channel();
  CHECK_NOT_EQUAL_RETURN(gamma_tensor->ElementsNum(), channel);
  CHECK_NOT_EQUAL_RETURN(beta_tensor->ElementsNum(), channel);

  param_->batch_ = in_tensor->Batch();
  MS_CHECK_INT_MUL_NOT_OVERFLOW(in_tensor->Height(), in_tensor->Width(), RET_ERROR);
  param_->inner_size_ = in_tensor->Height() * in_tensor->Width();
  param_->channel_ = in_tensor->Channel();
  param_->op_parameter_.thread_num_ = MSMIN(UP_DIV(param_->channel_, C8NUM), op_parameter_->thread_num_);
  return RET_OK;
}

int InstanceNormCPUKernel::DoInstanceNorm(int task_id) const {
  int ret = 0;
  if (input_pack_to_nc4hw4_) {  // arm64 x86-avx x86-sse x86
#ifdef ENABLE_AVX
    ret = InstanceNormNC8HW8(tmp_src_data_, dst_data_, gamma_data_, beta_data_, param_, task_id);
#else
    ret = InstanceNormNC4HW4(tmp_src_data_, dst_data_, gamma_data_, beta_data_, param_, task_id);
#endif
  } else {
    ret = InstanceNorm(tmp_src_data_, dst_data_, gamma_data_, beta_data_, param_, task_id);
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoInstanceNorm error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int InstanceNormRun(void *cdata, int task_id, float, float) {
  auto kernel = reinterpret_cast<const InstanceNormCPUKernel *>(cdata);
  auto ret = kernel->DoInstanceNorm(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InstanceNormRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int InstanceNormCPUKernel::Run() {
  src_data_ = reinterpret_cast<float *>(in_tensors_.at(FIRST_INPUT)->data());
  gamma_data_ = reinterpret_cast<float *>(in_tensors_.at(SECOND_INPUT)->data());
  beta_data_ = reinterpret_cast<float *>(in_tensors_.at(THIRD_INPUT)->data());
  dst_data_ = reinterpret_cast<float *>(out_tensors_.at(FIRST_INPUT)->data());
  CHECK_NULL_RETURN(src_data_);
  CHECK_NULL_RETURN(gamma_data_);
  CHECK_NULL_RETURN(beta_data_);
  CHECK_NULL_RETURN(dst_data_);
  if (in_tensors_[0]->format() == NC4HW4) {
#if defined(ENABLE_AVX) || defined(ENABLE_ARM64)
    tmp_src_data_ = src_data_;
#else  // other platform is not support nc4hw4 and must be pack to nc4hw4
    tmp_src_data_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(in_tensors_[0]->Size()));
    CHECK_NULL_RETURN(tmp_src_data_);
    PackNHWCToNC4HW4NotAlignedFp32(src_data_, tmp_src_data_, param_->batch_, param_->inner_size_, param_->channel_);
#endif
    input_pack_to_nc4hw4_ = true;
  } else if (in_tensors_[0]->format() == NHWC) {
    tmp_src_data_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(in_tensors_[0]->Size()));
    CHECK_NULL_RETURN(tmp_src_data_);
#ifdef ENABLE_AVX
    PackNHWCToNC8HW8NotAlignedFp32(src_data_, tmp_src_data_, param_->batch_, param_->inner_size_, param_->channel_);
#else
    PackNHWCToNC4HW4NotAlignedFp32(src_data_, tmp_src_data_, param_->batch_, param_->inner_size_, param_->channel_);
#endif
    input_pack_to_nc4hw4_ = true;
  } else {
    tmp_src_data_ = src_data_;
  }
  auto ret = ParallelLaunch(this->ms_context_, InstanceNormRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InstanceNormRun error error_code[" << ret << "]";
  }
  if (tmp_src_data_ != src_data_) {
    FreeTmpBuffer();
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_InstanceNorm, LiteKernelCreator<InstanceNormCPUKernel>)
}  // namespace mindspore::kernel
