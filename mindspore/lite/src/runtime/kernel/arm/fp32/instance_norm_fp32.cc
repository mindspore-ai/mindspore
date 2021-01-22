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
#include "src/runtime/kernel/arm/fp32/instance_norm_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp32/instance_norm_fp32.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_InstanceNorm;

namespace mindspore::kernel {
int InstanceNormCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int InstanceNormCPUKernel::ReSize() {
  param_->op_parameter_.thread_num_ = context_->thread_num_;
  auto shape = in_tensors_.front()->shape();
  param_->batch_ = shape[0];
  param_->inner_size_ = shape[2] * shape[3];
  param_->channel_ = shape[1];
  return RET_OK;
}

int InstanceNormCPUKernel::DoInstanceNorm(int task_id) {
  int ret = InstanceNorm(src_data_, dst_data_, gamma_data_, beta_data_, param_, task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoInstanceNorm error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int InstanceNormRun(void *cdata, int task_id) {
  auto kernel = reinterpret_cast<InstanceNormCPUKernel *>(cdata);
  auto ret = kernel->DoInstanceNorm(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InstanceNormRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int InstanceNormCPUKernel::Run() {
  src_data_ = reinterpret_cast<float *>(in_tensors_.at(0)->data_c());
  gamma_data_ = reinterpret_cast<float *>(in_tensors_.at(1)->data_c());
  beta_data_ = reinterpret_cast<float *>(in_tensors_.at(2)->data_c());
  dst_data_ = reinterpret_cast<float *>(out_tensors_.at(0)->data_c());
  auto ret = ParallelLaunch(this->context_->thread_pool_, InstanceNormRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InstanceNormRun error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_InstanceNorm, LiteKernelCreator<InstanceNormCPUKernel>)
}  // namespace mindspore::kernel
