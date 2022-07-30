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

#include <map>
#include <string>
#include <set>
#include "src/litert/kernel/opencl/kernel/activation.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "src/litert/kernel/opencl/cl/activation.cl.inc"

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

int ActivationOpenCLKernel::CheckSpecsWithoutShape() {
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_1 || out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  if (GetActTypeString(type_).empty()) {
    MS_LOG(WARNING) << "schema::ActivationType:" << type_ << "not found";
    return RET_ERROR;
  }
  return RET_OK;
}

int ActivationOpenCLKernel::CheckSpecs() { return RET_OK; }

int ActivationOpenCLKernel::Prepare() {
  out_shape_ = GpuTensorInfo::CreateGpuTensorInfo(out_tensors_[0]);
  if (out_shape_ == nullptr) {
    MS_LOG(ERROR) << "Create gpu tensor info failed.";
    return RET_ERROR;
  }
  std::string source = activation_source;
  const std::string program_name = "Activation";
  if (!ocl_runtime_->LoadSource(program_name, source)) {
    MS_LOG(ERROR) << "Load source failed.";
    return RET_ERROR;
  }
  const std::string kernel_name = GetActTypeString(type_);
  auto build_options_ext = CreateBuildOptionsExtByDType(this->registry_data_type_);
  auto ret = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options_ext);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build kernel failed.";
    return ret;
  }
  if (SetConstArgs() != RET_OK) {
    MS_LOG(ERROR) << "SeConstArgs failed.";
    return RET_ERROR;
  }
  (void)SetGlobalLocal();

  MS_LOG(DEBUG) << kernel_name << " init Done!";
  return RET_OK;
}

int ActivationOpenCLKernel::SetConstArgs() {
  int arg_idx = CLARGSINDEX2;
  cl_int2 image_size = {static_cast<int>(out_shape_->width), static_cast<int>(out_shape_->height)};
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, image_size) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (type_ == ActivationType_LEAKY_RELU) {
    if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, alpha_) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  }
  if (type_ == ActivationType_SIGMOID) {
    int c4 = out_shape_->Slice;
    int last_c4 = out_shape_->C % C4NUM == 0 ? C4NUM : out_shape_->C % C4NUM;
    if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, c4) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
    if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, last_c4) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int ActivationOpenCLKernel::SetGlobalLocal() {
  local_size_ = {};
  global_size_ = {out_shape_->width, out_shape_->height};
  AlignGlobalLocal(global_size_, local_size_);
  return RET_OK;
}

int ActivationOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  int arg_idx = CLARGSINDEX0;
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
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
