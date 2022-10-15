/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "src/litert/kernel/cpu/fp32/exp_fp32.h"
#include <cmath>
#include "include/errorcode.h"
#include "src/litert/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ExpFusion;

namespace mindspore::kernel {
int ExpCPUKernel::Prepare() {
  CHECK_NOT_EQUAL_RETURN(in_tensors_.size(), 1);
  CHECK_NOT_EQUAL_RETURN(out_tensors_.size(), 1);
  float log_base = (param_->base_ == -1) ? 1 : logf(param_->base_);
  param_->in_scale_ = param_->scale_ * log_base;
  if (param_->shift_ == 0) {
    param_->out_scale_ = 1;
  } else {
    if (log_base == 1) {
      param_->out_scale_ = expf(param_->shift_);
    } else {
      param_->out_scale_ = powf(param_->base_, param_->shift_);
    }
  }
  param_->op_parameter_.thread_num_ = ms_context_->thread_num_;
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ExpCPUKernel::ReSize() {
  param_->element_num_ = in_tensors_.front()->ElementsNum();
  return RET_OK;
}

int ExpCPUKernel::DoExcute(int task_id) {
  auto ret =
    ExpFusionFp32(reinterpret_cast<float *>(input_addr_), reinterpret_cast<float *>(output_addr_), param_, task_id);
  return ret;
}

int ExpRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto ExpData = reinterpret_cast<ExpCPUKernel *>(cdata);
  auto ret = ExpData->DoExcute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ExpRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ExpCPUKernel::Run() {
  input_addr_ = in_tensors_.front()->data();
  output_addr_ = out_tensors_.front()->data();
  CHECK_NULL_RETURN(input_addr_);
  CHECK_NULL_RETURN(output_addr_);

  auto ret = ParallelLaunch(this->ms_context_, ExpRun, this, ms_context_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Exp error: error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ExpFusion, LiteKernelCreator<ExpCPUKernel>)
}  // namespace mindspore::kernel
