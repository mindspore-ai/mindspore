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

#include <vector>
#include <map>
#include <string>
#include <set>

#include "src/runtime/kernel/opencl/kernel/activation.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"
#include "nnacl/fp32/common_func.h"
#include "src/runtime/kernel/opencl/cl/activation.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType_LEAKY_RELU;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::ActivationType_SIGMOID;
using mindspore::schema::ActivationType_TANH;
using mindspore::schema::PrimitiveType_Activation;

namespace mindspore::kernel {

int ActivationOpenClKernel::Init() {
  std::map<int, std::string> kernel_names{{ActivationType_LEAKY_RELU, "LeakyRelu"},
                                          {ActivationType_RELU, "Relu"},
                                          {ActivationType_SIGMOID, "Sigmoid"},
                                          {ActivationType_RELU6, "Relu6"},
                                          {ActivationType_TANH, "Tanh"}};
  if (kernel_names.count(type_) == 0) {
    MS_LOG(ERROR) << "schema::ActivationType:" << type_ << "not found";
    return mindspore::lite::RET_ERROR;
  }
  outShape = Image2DInfo(out_tensors_[0]);
  local_size_ = {};
  global_size_ = {outShape.width, outShape.height};
  std::string source = activation_source;
  std::set<std::string> build_options;
  std::string program_name = "Activation";
  ocl_runtime_->LoadSource(program_name, source);
  std::string kernel_name = kernel_names[type_];
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);
  SetArgs();
  MS_LOG(DEBUG) << kernel_name << " init Done!";
  return mindspore::lite::RET_OK;
}

int ActivationOpenClKernel::SetArgs() {
  int arg_idx = 2;
  cl_int2 image_size = {static_cast<int>(outShape.width), static_cast<int>(outShape.height)};
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, image_size);
  if (type_ == ActivationType_LEAKY_RELU) {
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, alpha_);
  }
  return RET_OK;
}

int ActivationOpenClKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " begin running!";
  int arg_idx = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->data_c());
  auto ret = ocl_runtime_->RunKernel(kernel_, global_size_, local_size_, nullptr);
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "Run kernel:" << this->name() << " fail.";
    return mindspore::lite::RET_ERROR;
  }
  return mindspore::lite::RET_OK;
}

kernel::LiteKernel *OpenClActivationKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                  const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                                  const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                                  const mindspore::lite::PrimitiveC *primitive) {
  if (inputs.empty()) {
    MS_LOG(ERROR) << "Input data size must be greater than 0, but your size is " << inputs.size();
    return nullptr;
  }
  if (inputs[0]->shape().size() > 2 && inputs[0]->shape()[0] > 1) {
    MS_LOG(ERROR) << "Activation kernel:" << opParameter->name_ << " failed: Unsupported multi-batch.";
    free(opParameter);
    return nullptr;
  }
  auto *kernel =
    new (std::nothrow) ActivationOpenClKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "New kernel:" << opParameter->name_ << "is nullptr.";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "Init activation kernel:" << opParameter->name_ << " failed!";
    delete kernel;
    return nullptr;
  }
  return kernel;
}
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Activation, OpenClActivationKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Activation, OpenClActivationKernelCreator)
}  // namespace mindspore::kernel
