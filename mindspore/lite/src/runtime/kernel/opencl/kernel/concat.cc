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
#include "src/runtime/kernel/opencl/kernel/concat.h"
#include <string>
#include <set>
#include "src/kernel_registry.h"
#include "src/runtime/opencl/opencl_runtime.h"
#include "src/runtime/kernel/arm/opclib/concat_parameter.h"
#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/fp32/concat.cl.inc"
#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Concat;

namespace mindspore::kernel {

int ConcatOpenCLKernel::Init() {
  if (inputs_[0]->shape().size() != 4) {
    MS_LOG(ERROR) << "only support dim=4";
  }

  auto param = reinterpret_cast<ConcatParameter *>(this->opParameter);
  MS_LOG(DEBUG) << "concat at axis=:  " << param->axis_;
  if (param->axis_ != 0 && param->axis_ != 3) {
    MS_LOG(ERROR) << "only support axis=0 or axis=3";
  }

  if (param->axis_ == 0) {
    return 0;
  }

  std::string kernel_name = "Concat";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
#ifdef PROGRAM_WITH_IL
  ocl_runtime->CreateKernelFromIL(kernel_(), kernel_name);
#else
  std::set<std::string> build_options;
  std::string source = concat_source_fp32;
  std::string program_name = "Concat";
  ocl_runtime->LoadSource(program_name, source);
  ocl_runtime->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif
  outputs_[0]->SetFormat(schema::Format_NHWC4);
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return 0;
}

int ConcatOpenCLKernel::ReSize() { return 0; }

int ConcatOpenCLKernel::Run_axis0() {
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  auto allocator_ = ocl_runtime->GetAllocator();
  cl::CommandQueue *command_queue = ocl_runtime->GetDefaultCommandQueue();

  for (auto &tensor : inputs_) {
    auto buffer = static_cast<cl::Buffer *>(allocator_->GetDeviceBuffer(tensor->Data()));
    ocl_runtime->MapBuffer(*buffer, CL_MAP_READ, tensor->Size(), command_queue, true);
  }
  for (auto &tensor : outputs_) {
    auto buffer = static_cast<cl::Buffer *>(allocator_->GetDeviceBuffer(tensor->Data()));
    ocl_runtime->MapBuffer(*buffer, CL_MAP_WRITE, tensor->Size(), command_queue, true);
  }

  memcpy_s(outputs_[0]->Data(), inputs_[0]->Size(), inputs_[0]->Data(), inputs_[0]->Size());
  memcpy_s(reinterpret_cast<char *>(outputs_[0]->Data()) + inputs_[0]->Size(), inputs_[1]->Size(), inputs_[1]->Data(),
           inputs_[1]->Size());

  for (auto tensors : {&inputs_, &outputs_}) {
    for (auto &tensor : *tensors) {
      auto buffer = static_cast<cl::Buffer *>(allocator_->GetDeviceBuffer(tensor->Data()));
      ocl_runtime->UnmapBuffer(*buffer, tensor->Data());
    }
  }
  return 0;
}

int ConcatOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->Name() << " Running!";
  auto param = reinterpret_cast<ConcatParameter *>(this->opParameter);
  if (param->axis_ == 0) {
    return Run_axis0();
  }

  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  std::vector<size_t> local = {1, 1, 1};
  std::vector<size_t> global = {1, 1, 1};

  auto input0_shape = inputs_[0]->shape();
  auto input1_shape = inputs_[1]->shape();
  auto output_shape = outputs_[0]->shape();
  cl_int4 input0_shape_ = {input0_shape[0], input0_shape[1], input0_shape[2], input0_shape[3]};
  cl_int4 input1_shape_ = {input1_shape[0], input1_shape[1], input1_shape[2], input1_shape[3]};
  cl_int4 output_shape_ = {output_shape[0], output_shape[1], output_shape[2], output_shape[3]};

  int arg_cn = 0;
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, inputs_[0]->Data());
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, inputs_[1]->Data());
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, outputs_[0]->Data());
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, input0_shape_);
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, input1_shape_);
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, output_shape_);
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, param->axis_);

  ocl_runtime->RunKernel(kernel_, global, local, nullptr);
  return 0;
}

kernel::LiteKernel *OpenCLConcatKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                              const std::vector<lite::tensor::Tensor *> &outputs,
                                              OpParameter *opParameter, const lite::Context *ctx,
                                              const kernel::KernelKey &desc) {
  auto *kernel = new ConcatOpenCLKernel(opParameter, inputs, outputs);
  auto ret = kernel->Init();
  if (0 != ret) {
    MS_LOG(ERROR) << "Init kernel failed, name: Convolution";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Concat, OpenCLConcatKernelCreator);
}  // namespace mindspore::kernel

