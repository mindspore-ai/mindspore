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
#include "nnacl/fp32/common_func_fp32.h"
#include "src/runtime/kernel/opencl/cl/activation.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType_HSIGMOID;
using mindspore::schema::ActivationType_HSWISH;
using mindspore::schema::ActivationType_LEAKY_RELU;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::ActivationType_SIGMOID;
using mindspore::schema::ActivationType_SWISH;
using mindspore::schema::ActivationType_TANH;
using mindspore::schema::PrimitiveType_Activation;

namespace mindspore::kernel {

std::string ActivationOpenCLKernel::GetActTypeString(int act_type) {
  static std::map<int, std::string> supported_act_type = {
    {ActivationType_LEAKY_RELU, "LeakyRelu"}, {ActivationType_RELU, "Relu"},        {ActivationType_SIGMOID, "Sigmoid"},
    {ActivationType_RELU6, "Relu6"},          {ActivationType_TANH, "Tanh"},        {ActivationType_SWISH, "Swish"},
    {ActivationType_HSWISH, "HSwish"},        {ActivationType_HSIGMOID, "HSigmoid"}};
  auto result_iter = supported_act_type.find(act_type);
  if (result_iter != supported_act_type.end()) {
    return result_iter->second;
  }
  return "";
}

int ActivationOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() != 1 || out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  if (GetActTypeString(type_).empty()) {
    MS_LOG(ERROR) << "schema::ActivationType:" << type_ << "not found";
    return RET_ERROR;
  }
  return RET_OK;
}

int ActivationOpenCLKernel::Prepare() {
  outShape = GpuTensorInfo(out_tensors_[0]);
  std::string source = activation_source;
  std::string program_name = "Activation";
  ocl_runtime_->LoadSource(program_name, source);
  std::string kernel_name = GetActTypeString(type_);
  auto build_options_ext = CreateBuildOptionsExtByDType(desc_.data_type);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options_ext);
  SetConstArgs();
  SetGlobalLocal();
  MS_LOG(DEBUG) << kernel_name << " init Done!";
  return mindspore::lite::RET_OK;
}

void ActivationOpenCLKernel::SetConstArgs() {
  int arg_idx = 2;
  cl_int2 image_size = {static_cast<int>(outShape.width), static_cast<int>(outShape.height)};
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, image_size);
  if (type_ == ActivationType_LEAKY_RELU) {
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, alpha_);
  }
  if (type_ == ActivationType_SIGMOID) {
    int c4 = outShape.Slice;
    int last_c4 = outShape.C % 4 == 0 ? 4 : outShape.C % 4;
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, c4);
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, last_c4);
  }
}

void ActivationOpenCLKernel::SetGlobalLocal() {
  local_size_ = {};
  global_size_ = {outShape.width, outShape.height};
  AlignGlobalLocal(global_size_, local_size_);
}

int ActivationOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  int arg_idx = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->data_c());
  auto ret = ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run kernel:" << this->name() << " fail.";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Activation, OpenCLKernelCreator<ActivationOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Activation, OpenCLKernelCreator<ActivationOpenCLKernel>)
}  // namespace mindspore::kernel
