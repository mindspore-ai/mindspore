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

#include "src/runtime/kernel/arm/fp32/activation.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"
#include "src/ops/ops.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType_HSWISH;
using mindspore::schema::ActivationType_LEAKY_RELU;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::PrimitiveType_Activation;

namespace mindspore::kernel {
int ActivationCPUKernel::Init() { return RET_OK; }

int ActivationCPUKernel::ReSize() { return RET_OK; }

int ActivationCPUKernel::DoActivation(int task_id) {
  auto input_addr = reinterpret_cast<float *>(in_tensors_.at(0)->Data());
  auto output_addr = reinterpret_cast<float *>(out_tensors_.at(0)->Data());
  auto length = in_tensors_.at(0)->ElementsNum();

  int stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);

  auto error_code = RET_OK;

  if (type_ == schema::ActivationType_RELU) {
    error_code = Fp32Relu(input_addr + stride * task_id, count, output_addr + stride * task_id);
  } else if (type_ == schema::ActivationType_RELU6) {
    error_code = Fp32Relu6(input_addr + stride * task_id, count, output_addr + stride * task_id);
  } else if (type_ == schema::ActivationType_LEAKY_RELU) {
    error_code = LRelu(input_addr + stride * task_id, count, output_addr + stride * task_id, alpha_);
  } else if (type_ == schema::ActivationType_SIGMOID) {
    error_code = Sigmoid(input_addr + stride * task_id, count, output_addr + stride * task_id);
  } else if (type_ == schema::ActivationType_TANH) {
    error_code = Tanh(input_addr + stride * task_id, count, output_addr + stride * task_id);
  } else if (type_ == schema::ActivationType_HSWISH) {
    error_code = HSwish(input_addr + stride * task_id, count, output_addr + stride * task_id);
  } else {
    MS_LOG(ERROR) << "Activation type error";
    return RET_ERROR;
  }
  if (error_code != RET_OK) {
    return RET_ERROR;
  }
  return RET_OK;
}

int ActivationRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto activation_kernel = reinterpret_cast<ActivationCPUKernel *>(cdata);
  auto error_code = activation_kernel->DoActivation(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ActivationRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ActivationCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return ret;
  }
  int error_code = LiteBackendParallelLaunch(ActivationRun, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Activation function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuActivationFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                   const std::vector<lite::tensor::Tensor *> &outputs,
                                                   OpParameter *opParameter, const lite::Context *ctx,
                                                   const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Activation);
  auto *kernel = new (std::nothrow) ActivationCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Activation, CpuActivationFp32KernelCreator)
}  // namespace mindspore::kernel
