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

#include "src/runtime/kernel/arm/fp16_grad/activation_fp16_grad.h"
#include "nnacl/fp16_grad/activation_grad.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_SIGMOID;
using mindspore::schema::PrimitiveType_ActivationGrad;

namespace mindspore::kernel {
int ActivationGradCPUKernelFp16::Init() {
  if (in_tensors_.size() != 2) {
    MS_LOG(ERROR) << "ActivationGrad should have 2 input tensors";
    return RET_ERROR;
  }
  CHECK_LESS_RETURN(in_tensors_.size(), 2);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_.at(0));
  CHECK_NULL_RETURN(in_tensors_.at(1));
  CHECK_NULL_RETURN(out_tensors_.at(0));
  CHECK_NULL_RETURN(param_act_grad_);
  return RET_OK;
}

int ActivationGradCPUKernelFp16::ReSize() { return RET_OK; }

int ActivationGradCPUKernelFp16::DoActivation(int task_id) {
  auto yt_addr = reinterpret_cast<float16_t *>(in_tensors_.at(0)->MutableData());
  CHECK_NULL_RETURN(yt_addr);
  auto input_addr = reinterpret_cast<float16_t *>(in_tensors_.at(1)->MutableData());
  CHECK_NULL_RETURN(input_addr);
  auto output_addr = reinterpret_cast<float16_t *>(out_tensors_.at(0)->MutableData());
  CHECK_NULL_RETURN(output_addr);
  int length = in_tensors_.at(0)->ElementsNum();

  int stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);
  int start = stride * task_id;

  auto error_code = RET_OK;

  if (param_act_grad_->type_ == schema::ActivationType_RELU) {
    error_code = ReluFp16Grad(yt_addr + start, input_addr + start, count, output_addr + start);
  } else if (param_act_grad_->type_ == schema::ActivationType_RELU6) {
    error_code = Relu6Fp16Grad(yt_addr + start, input_addr + start, count, output_addr + start);
  } else if (param_act_grad_->type_ == schema::ActivationType_LEAKY_RELU) {
    error_code = LReluFp16Grad(yt_addr + start, input_addr + start, count, output_addr + start,
                               (float16_t)param_act_grad_->alpha_);
  } else if (param_act_grad_->type_ == schema::ActivationType_SIGMOID) {
    // Sigmoid gets the input tensors in reverse order!
    error_code = SigmoidFp16Grad(input_addr + start, yt_addr + start, count, output_addr + start);
  } else if (param_act_grad_->type_ == schema::ActivationType_TANH) {
    error_code = TanhFp16Grad(yt_addr + start, input_addr + start, count, output_addr + start);
  } else if (param_act_grad_->type_ == schema::ActivationType_HSWISH) {
    error_code = HSwishFp16Grad(yt_addr + start, input_addr + start, count, output_addr + start);
  } else if (param_act_grad_->type_ == schema::ActivationType_HSIGMOID) {
    error_code = HSigmoidFp16Grad(yt_addr + start, input_addr + start, count, output_addr + start);
  } else if (param_act_grad_->type_ == schema::ActivationType_ELU) {
    error_code =
      EluFp16Grad(yt_addr + start, input_addr + start, count, output_addr + start, (float16_t)param_act_grad_->alpha_);
  } else if (param_act_grad_->type_ == schema::ActivationType_GELU) {
    error_code = GeluFp16Grad(yt_addr + start, input_addr + start, count, output_addr + start);
  } else {
    MS_LOG(ERROR) << "Activation type error";
    return RET_ERROR;
  }
  if (error_code != RET_OK) {
    return RET_ERROR;
  }
  return RET_OK;
}

int ActivationGradRunFp16(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto activationGrad_kernel = reinterpret_cast<ActivationGradCPUKernelFp16 *>(cdata);
  auto error_code = activationGrad_kernel->DoActivation(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ActivationGradRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ActivationGradCPUKernelFp16::Run() {
  int error_code = ParallelLaunch(this->ms_context_, ActivationGradRunFp16, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Activation Grad function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_ActivationGrad, LiteKernelCreator<ActivationGradCPUKernelFp16>)
}  // namespace mindspore::kernel
