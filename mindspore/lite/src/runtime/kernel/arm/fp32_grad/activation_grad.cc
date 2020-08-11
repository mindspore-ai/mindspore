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
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationGradType_HSWISH;
using mindspore::schema::ActivationGradType_LEAKY_RELU;
using mindspore::schema::ActivationGradType_RELU;
using mindspore::schema::ActivationGradType_RELU6;
using mindspore::schema::PrimitiveType_ActivationGrad;

namespace mindspore::kernel {
int ActivationGradCPUKernel::Init() {
  outputs_[0]->set_shape(inputs_[0]->shape());
  return RET_OK;
}

int ActivationGradCPUKernel::ReSize() { return RET_OK; }

int ActivationGradCPUKernel::DoActivation(int task_id) {
  auto yt_addr = reinterpret_cast<float *>(inputs_.at(0)->Data());
  auto input_addr = reinterpret_cast<float *>(inputs_.at(1)->Data());
  auto output_addr = reinterpret_cast<float *>(outputs_.at(0)->Data());
  auto length = inputs_.at(0)->ElementsNum();

  auto error_code = RET_OK;

  if (type_ == schema::ActivationGradType_RELU) {
    error_code = ReluGrad(yt_addr, input_addr, length, output_addr);
  } else if (type_ == schema::ActivationGradType_RELU6) {
    error_code = Relu6Grad(yt_addr, input_addr, length, output_addr);
  } else if (type_ == schema::ActivationGradType_LEAKY_RELU) {
    error_code = LReluGrad(yt_addr, input_addr, length, output_addr, alpha_);
  } else if (type_ == schema::ActivationGradType_SIGMOID) {
    error_code = SigmoidGrad(yt_addr, input_addr, length, output_addr);
  } else if (type_ == schema::ActivationGradType_TANH) {
    error_code = TanhGrad(yt_addr, input_addr, length, output_addr);
  } else if (type_ == schema::ActivationGradType_HSWISH) {
    error_code = HSwishGrad(yt_addr, input_addr, length, output_addr);
  } else if (type_ == schema::ActivationGradType_HSIGMOID) {
    error_code = HSigmoidGrad(yt_addr, input_addr, length, output_addr);
  } else {
    MS_LOG(ERROR) << "Activation type error";
    return RET_ERROR;
  }
  if (error_code != RET_OK) {
    return RET_ERROR;
  }
  return RET_OK;
}

int ActivationGradRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto activationGrad_kernel = reinterpret_cast<ActivationGradCPUKernel *>(cdata);
  auto error_code = activationGrad_kernel->DoActivation(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ActivationGradRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ActivationGradCPUKernel::Run() {
  int error_code = LiteBackendParallelLaunch(ActivationGradRun, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Activation function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuActivationGradFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                       const std::vector<lite::tensor::Tensor *> &outputs,
                                                       OpParameter *opParameter, const lite::Context *ctx,
                                                       const kernel::KernelKey &desc,
                                                       const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_ActivationGrad);
  auto *kernel = new (std::nothrow) ActivationGradCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  MS_ASSERT(kernel != nullptr);
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InferShape kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ActivationGrad, CpuActivationGradFp32KernelCreator)
}  // namespace mindspore::kernel
