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

#include "src/runtime/kernel/opencl/kernel/arithmetic.h"
#include <set>
#include <vector>
#include <string>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/utils.h"
#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/fp32/arithmetic_buffer.cl.inc"
#include "src/runtime/kernel/opencl/cl/fp32/arithmetic_image2d.cl.inc"
#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;

namespace mindspore::kernel {

std::vector<size_t> ArithmeticOpenCLKernel::InitGlobalSize() const {
  const size_t global_x = outputs_[0]->Width();
  const size_t global_y = outputs_[0]->Height();
  const size_t global_z = UP_ROUND_DIV(outputs_[0]->Channel(), 4);
  std::vector<size_t> global = {global_x, global_y, global_z};
  return global;
}

void ArithmeticOpenCLKernel::Image2dGetWorkGroupSize() {
  global_size_ = InitGlobalSize();
  int max_work_group_size = runtime_->GetKernelMaxWorkGroupSize(kernel_(), (*runtime_->Device())());
  local_size_ = GetCommonLocalSize(global_size_, max_work_group_size);
  global_size_ = GetCommonGlobalSize(local_size_, global_size_);
}

void ArithmeticOpenCLKernel::BufferGetWorkGroupSize() {
  uint32_t element_num = outputs_[0]->ElementsC4Num();
  global_size_ = {element_num};
}

int ArithmeticOpenCLKernel::Init() {
  runtime_ = lite::opencl::OpenCLRuntime::GetInstance();
  std::string element_name;
  std::string boardcast_name;

  if (inputs_[1]->TensorType() == schema::NodeType_ValueNode && inputs_[1]->Data() != nullptr) {
    element_flag_ = false;
  } else {
    element_flag_ = true;
  }

  switch (opParameter->type_) {
    case PrimitiveType_Mul:
      element_name = "ElementMul";
      boardcast_name = "BoardcastMul";
      break;
    case PrimitiveType_Add:
      element_name = "ElementAdd";
      boardcast_name = "BoardcastAdd";
      break;
    case PrimitiveType_Sub:
      element_name = "ElementSub";
      boardcast_name = "BoardcastSub";
      break;
    case PrimitiveType_Div:
      element_name = "ElementDiv";
      boardcast_name = "BoardcastDiv";
      break;
    default:
      MS_LOG(ERROR) << "Error Operator type " << opParameter->type_;
      break;
  }

#ifdef PROGRAM_WITH_IL
  runtime_->CreateKernelFromIL(kernel_(), kernel_name);
#else
  std::string program_name = "Arithmetic";
  std::set<std::string> build_options;
  std::string source = arithmetic_buffer_source_fp32;
  runtime_->LoadSource(program_name, source);

  if (element_flag_) {
    runtime_->BuildKernel(kernel_, program_name, element_name, build_options);
    MS_LOG(DEBUG) << element_name << " Init Done!";
  } else {
    runtime_->BuildKernel(kernel_, program_name, boardcast_name, build_options);
    MS_LOG(DEBUG) << boardcast_name << " Init Done!";
  }
#endif
  outputs_[0]->SetFormat(schema::Format_NHWC4);
  return 0;
}

int ArithmeticOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->Name() << " Running!";
  auto runtime_ = lite::opencl::OpenCLRuntime::GetInstance();
  BufferGetWorkGroupSize();

  int arg_idx = 0;
  uint32_t element_num = outputs_[0]->ElementsC4Num();

  runtime_->SetKernelArg(kernel_, arg_idx++, inputs_[0]->Data());
  if (element_flag_) {
    runtime_->SetKernelArg(kernel_, arg_idx++, inputs_[1]->Data());
  } else {
    runtime_->SetKernelArg(kernel_, arg_idx++, static_cast<float *>(inputs_[1]->Data())[0]);
  }
  runtime_->SetKernelArg(kernel_, arg_idx++, outputs_[0]->Data());
  runtime_->SetKernelArg(kernel_, arg_idx++, element_num);

  runtime_->RunKernel(kernel_, global_size_, local_size_, nullptr);
  return 0;
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

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Mul, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Add, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Sub, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Div, OpenCLArithmeticKernelCreator)
}  // namespace mindspore::kernel
