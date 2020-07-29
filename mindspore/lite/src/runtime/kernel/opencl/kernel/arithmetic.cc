/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/opencl/kernel/arithmetic.h"
#include <vector>
#include <string>
#include <set>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/fp32/arithmetic.cl.inc"
#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;

namespace mindspore::kernel {

int ArithmeticOpenCLKernel::Init() {
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  std::string kernel_name = "ArithmeticAdd";

  is_bias_add_ = false;
  if (inputs_[1]->TensorType() == schema::NodeType_ValueNode && inputs_[1]->Data() != nullptr) {
    kernel_name = "ArithmeticBiasAdd";
    is_bias_add_ = true;
  }

  switch (opParameter->type_) {
    case PrimitiveType_Mul:
      if (is_bias_add_) {
        weight_ = static_cast<float *>(inputs_[1]->Data())[0];
        break;
      }
      kernel_name = "ArithmeticMul";
      break;
    case PrimitiveType_Add:
      if (is_bias_add_) {
        bias_ = static_cast<float *>(inputs_[1]->Data())[0];
        break;
      }
      kernel_name = "ArithmeticAdd";
      break;
    case PrimitiveType_Sub:
      if (is_bias_add_) {
        bias_ = -1 * static_cast<float *>(inputs_[1]->Data())[0];
        break;
      }
      kernel_name = "ArithmeticSub";
      break;
    case PrimitiveType_Div:
      if (is_bias_add_) {
        weight_ = 1 / static_cast<float *>(inputs_[1]->Data())[0];
        break;
      }
      kernel_name = "ArithmeticDiv";
      break;
    default:
      MS_LOG(ERROR) << "Error Operator type " << opParameter->type_;
      break;
  }

#ifdef PROGRAM_WITH_IL
  ocl_runtime->CreateKernelFromIL(kernel_(), kernel_name);
#else
  std::string program_name = "Arithmetic";
  std::set<std::string> build_options;
  std::string source = arithmetic_source_fp32;
  ocl_runtime->LoadSource(program_name, source);
  ocl_runtime->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif
  outputs_[0]->SetFormat(schema::Format_NHWC4);
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return 0;
}

int ArithmeticOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->Name() << " Running!";
  uint32_t element_num = outputs_[0]->ElementsC4Num();
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  std::vector<size_t> global = {element_num};
  std::vector<size_t> local;

  ocl_runtime->SetKernelArg(kernel_, 0, inputs_[0]->Data());
  if (is_bias_add_) {
    MS_LOG(DEBUG) << "weight: " << weight_ << " bias: " << bias_;
    ocl_runtime->SetKernelArg(kernel_, 1, outputs_[0]->Data());
    ocl_runtime->SetKernelArg(kernel_, 2, weight_);
    ocl_runtime->SetKernelArg(kernel_, 3, bias_);
    ocl_runtime->SetKernelArg(kernel_, 4, element_num / C4NUM);
  } else {
    ocl_runtime->SetKernelArg(kernel_, 1, inputs_[1]->Data());
    ocl_runtime->SetKernelArg(kernel_, 2, outputs_[0]->Data());
    ocl_runtime->SetKernelArg(kernel_, 3, element_num);
  }
  return ocl_runtime->RunKernel(kernel_, global, local, nullptr);
}

kernel::LiteKernel *OpenCLArithmeticKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                  const std::vector<lite::tensor::Tensor *> &outputs,
                                                  OpParameter *opParameter, const lite::Context *ctx,
                                                  const kernel::KernelKey &desc) {
  auto *kernel = new ArithmeticOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create OpenCL Arithmetic kernel failed!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (0 != ret) {
    MS_LOG(ERROR) << "Init kernel failed, name: Arithmetic";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, PrimitiveType_Mul, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, PrimitiveType_Add, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, PrimitiveType_Sub, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, PrimitiveType_Div, OpenCLArithmeticKernelCreator)
}  // namespace mindspore::kernel

