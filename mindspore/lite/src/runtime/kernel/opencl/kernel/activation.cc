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
#include <string>
#include <set>

#include "src/runtime/kernel/opencl/kernel/activation.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"
#include "src/ops/ops.h"
#include "src/runtime/kernel/opencl/cl/fp32/activation.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType_LEAKY_RELU;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::ActivationType_SIGMOID;
using mindspore::schema::PrimitiveType_Activation;

namespace mindspore::kernel {

int ActivationOpenClKernel::Init() {
  const int max_shape_dim = 4;
  if (in_tensors_[0]->shape().size() != max_shape_dim) {
    MS_LOG(ERROR) << "Activate fun only support dim=4, but your dim=" << in_tensors_[0]->shape().size();
    return RET_ERROR;
  }
  std::string program_name = "";
  std::string kernel_name = "";
  std::string source = activation_source_fp32;
  if (type_ == ActivationType_RELU) {
    program_name = "RELU";
    kernel_name = "Relu";
  } else if (type_ == ActivationType_RELU6) {
    program_name = "RELU6";
    kernel_name = "Relu6";
  } else if (type_ == ActivationType_LEAKY_RELU) {
    program_name = "LEAKY_RELU";
    kernel_name = "ReluScalar";
  } else if (type_ == ActivationType_SIGMOID) {
    program_name = "SIGMOID";
    kernel_name = "Sigmoid";
  } else {
    MS_LOG(ERROR) << "Activation type error";
    return RET_ERROR;
  }
  std::set<std::string> build_options;
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->LoadSource(program_name, source);
  ocl_runtime->BuildKernel(kernel_, program_name, kernel_name, build_options);
  MS_LOG(DEBUG) << op_parameter_->name_ << " init Done!";
  return RET_OK;
}

int ActivationOpenClKernel::Run() {
  MS_LOG(DEBUG) << op_parameter_->name_ << " begin running!";
  int N = in_tensors_[0]->shape()[0];
  int H = in_tensors_[0]->shape()[1];
  int W = in_tensors_[0]->shape()[2];
  int C = in_tensors_[0]->shape()[3];
  cl_int4 input_shape = {N, H, W, C};

  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  int arg_idx = 0;
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->Data());
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->Data());
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, input_shape);
  if (type_ == ActivationType_LEAKY_RELU) {
    ocl_runtime->SetKernelArg(kernel_, arg_idx++, alpha_);
  }
  std::vector<size_t> local = {1, 1};
  std::vector<size_t> global = {static_cast<size_t>(H), static_cast<size_t>(W)};
  std::cout << type_ << " " << std::endl;
  auto ret = ocl_runtime->RunKernel(kernel_, global, local, nullptr);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run kernel:" << op_parameter_->name_ << " fail.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ActivationOpenClKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  int H = in_tensors_[0]->shape()[1];
  int W = in_tensors_[0]->shape()[2];
  int C = in_tensors_[0]->shape()[3];

#ifdef ENABLE_FP16
  size_t img_dtype = CL_HALF_FLOAT;
#else
  size_t img_dtype = CL_FLOAT;
#endif

  img_size->clear();
  img_size->push_back(W * UP_DIV(C, C4NUM));
  img_size->push_back(H);
  img_size->push_back(img_dtype);
  return RET_OK;
}

kernel::LiteKernel *OpenClActivationFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                      const std::vector<lite::tensor::Tensor *> &outputs,
                                                      OpParameter *opParameter, const lite::Context *ctx,
                                                      const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  if (inputs.size() == 0) {
    MS_LOG(ERROR) << "Input data size must be greater than 0, but your size is " << inputs.size();
    return nullptr;
  }
  if (inputs[0]->shape()[0] > 1) {
    MS_LOG(ERROR) << "Activation kernel:" << opParameter->name_ << " failed: Unsupported multi-batch.";
    return nullptr;
  }
  auto *kernel =
    new (std::nothrow) ActivationOpenClKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "New kernel:" << opParameter->name_ << "is nullptr.";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init activation kernel:" << opParameter->name_ << " failed!";
    delete kernel;
    return nullptr;
  }
  return kernel;
}
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Activation, OpenClActivationFp32KernelCreator)
}  // namespace mindspore::kernel
