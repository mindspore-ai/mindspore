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

#include "src/runtime/kernel/arm/fp32_grad/activation_grad.h"
#include "nnacl/fp32_grad/activation_grad.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType_ELU;
using mindspore::schema::ActivationType_GELU;
using mindspore::schema::ActivationType_HSWISH;
using mindspore::schema::ActivationType_LEAKY_RELU;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::PrimitiveType_ActivationGrad;

namespace mindspore::kernel {
int ActivationGradCPUKernel::Init() {
  if (in_tensors_.size() < 2) {
    MS_LOG(ERROR) << "ActivationGrad should have more than 2 input tensors";
    return RET_ERROR;
  }
  return RET_OK;
}

int ActivationGradCPUKernel::ReSize() { return RET_OK; }

int ActivationGradCPUKernel::DoActivation(int task_id) {
  auto yt_addr = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  auto input_addr = reinterpret_cast<float *>(in_tensors_.at(1)->MutableData());
  auto output_addr = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  int length = in_tensors_.at(0)->ElementsNum();

  int stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);
  size_t start = stride * task_id;

  auto error_code = RET_OK;
  if (count > 0) {
    if (param_act_grad_->type_ == schema::ActivationType_RELU) {
      error_code = ReluGrad(yt_addr + start, input_addr + start, count, output_addr + start);
    } else if (param_act_grad_->type_ == schema::ActivationType_RELU6) {
      error_code = Relu6Grad(yt_addr + start, input_addr + start, count, output_addr + start);
    } else if (param_act_grad_->type_ == schema::ActivationType_LEAKY_RELU) {
      error_code = LReluGrad(yt_addr + start, input_addr + start, count, output_addr + start, param_act_grad_->alpha_);
    } else if (param_act_grad_->type_ == schema::ActivationType_SIGMOID) {
      // Sigmoid gets the input tensors in reverse order!
      error_code = SigmoidGrad(input_addr + start, yt_addr + start, count, output_addr + start);
    } else if (param_act_grad_->type_ == schema::ActivationType_TANH) {
      error_code = TanhGrad(yt_addr + start, input_addr + start, count, output_addr + start);
    } else if (param_act_grad_->type_ == schema::ActivationType_HSWISH) {
      error_code = HSwishGrad(yt_addr + start, input_addr + start, count, output_addr + start);
    } else if (param_act_grad_->type_ == schema::ActivationType_HSIGMOID) {
      error_code = HSigmoidGrad(yt_addr + start, input_addr + start, count, output_addr + start);
    } else if (param_act_grad_->type_ == schema::ActivationType_ELU) {
      error_code = EluGrad(yt_addr + start, input_addr + start, count, output_addr + start, param_act_grad_->alpha_);
    } else if (param_act_grad_->type_ == schema::ActivationType_GELU) {
      error_code = GeluGrad(yt_addr + start, input_addr + start, count, output_addr + start);
    } else {
      MS_LOG(ERROR) << "Activation type error";
      return RET_ERROR;
    }
    if (error_code != RET_OK) {
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int ActivationGradRun(void *cdata, int task_id) {
  MS_ASSERT(cdata != nullptr);
  auto activationGrad_kernel = reinterpret_cast<ActivationGradCPUKernel *>(cdata);
  auto error_code = activationGrad_kernel->DoActivation(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ActivationGradRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ActivationGradCPUKernel::Run() {
  int error_code = ParallelLaunch(this->context_->thread_pool_, ActivationGradRun, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Activation Grad function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ActivationGrad, LiteKernelCreator<ActivationGradCPUKernel>)
}  // namespace mindspore::kernel
