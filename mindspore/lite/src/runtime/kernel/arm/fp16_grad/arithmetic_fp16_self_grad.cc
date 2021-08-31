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

#include "src/runtime/kernel/arm/fp16_grad/arithmetic_fp16_self_grad.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LogGrad;

namespace mindspore::kernel {
int ArithmeticSelfGradFp16CPUKernel::Init() {
  if (in_tensors_.size() != 2) {
    MS_LOG(ERROR) << "ActivationGrad should have 2 input tensors";
    return RET_ERROR;
  }
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_.at(0));
  CHECK_NULL_RETURN(in_tensors_.at(1));
  CHECK_NULL_RETURN(out_tensors_.at(0));
  return RET_OK;
}

int ArithmeticSelfGradFp16CPUKernel::ReSize() { return RET_OK; }

int ArithmeticSelfGradFp16CPUKernel::DoActivation(int task_id) {
  auto yt_addr = reinterpret_cast<float16_t *>(in_tensors_.at(0)->MutableData());
  auto input_addr = reinterpret_cast<float16_t *>(in_tensors_.at(1)->MutableData());
  auto output_addr = reinterpret_cast<float16_t *>(out_tensors_.at(0)->MutableData());
  int length = in_tensors_.at(0)->ElementsNum();
  CHECK_NULL_RETURN(yt_addr);
  CHECK_NULL_RETURN(input_addr);
  CHECK_NULL_RETURN(output_addr);

  int stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);
  int start = stride * task_id;

  auto error_code = RET_OK;
  CHECK_NULL_RETURN(param_act_grad_);
  if (param_act_grad_->type_ == schema::PrimitiveType_LogGrad) {
    error_code = Fp16LogGrad(yt_addr + start, input_addr + start, count, output_addr + start);
  } else {
    MS_LOG(ERROR) << "Activation type error";
    return RET_ERROR;
  }
  if (error_code != RET_OK) {
    return RET_ERROR;
  }
  return RET_OK;
}

int ArithmeticSelfGradFp16Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto activationGrad_kernel = reinterpret_cast<ArithmeticSelfGradFp16CPUKernel *>(cdata);
  CHECK_NULL_RETURN(activationGrad_kernel);
  auto error_code = activationGrad_kernel->DoActivation(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ActivationGradRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ArithmeticSelfGradFp16CPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, ArithmeticSelfGradFp16Run, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Activation Grad function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_LogGrad, LiteKernelCreator<ArithmeticSelfGradFp16CPUKernel>)
}  // namespace mindspore::kernel
