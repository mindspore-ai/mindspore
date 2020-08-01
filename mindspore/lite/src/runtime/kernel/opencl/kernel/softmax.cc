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

#include "src/runtime/kernel/opencl/kernel/softmax.h"
#include <string>
#include <set>
#include "src/kernel_registry.h"
#include "src/runtime/opencl/opencl_runtime.h"
#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/fp32/softmax.cl.inc"
#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_SoftMax;

namespace mindspore {
namespace kernel {
int SoftmaxOpenCLKernel::Init() {
  std::string kernel_name = "SoftMax";
  if (parameter_->axis_ != -1 && parameter_->axis_ != 3) {
    MS_LOG(ERROR) << "Init `Softmax` kernel failed: Unsupported axis: " << parameter_->axis_;
    return -1;
  }

  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
#ifdef PROGRAM_WITH_IL
  ocl_runtime->CreateKernelFromIL(kernel_(), kernel_name);
#else
  std::set<std::string> build_options;
  std::string source = softmax_source_fp32;
  std::string program_name = "SoftMax";
  ocl_runtime->LoadSource(program_name, source);
  ocl_runtime->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif
  outputs_[0]->SetFormat(schema::Format_NHWC4);
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return 0;
}

int SoftmaxOpenCLKernel::InitBuffer() { return 0; }
int SoftmaxOpenCLKernel::ReSize() { return 0; }

int SoftmaxOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->Name() << " Running!";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  auto allocator = ocl_runtime->GetAllocator();

  // global and local workers
  const uint32_t grid_x = inputs_[0]->shape()[2];  // W
  const uint32_t grid_y = inputs_[0]->shape()[1];  // H
  const uint32_t grid_z = 1;
  std::vector<size_t> global = {grid_x, grid_y, grid_z};
  std::vector<size_t> local = {1, 1, 1};

  // input and output
  cl::Buffer *input = reinterpret_cast<cl::Buffer *>(allocator->GetDeviceBuffer(inputs_[0]->Data()));
  cl::Buffer *output = reinterpret_cast<cl::Buffer *>(allocator->GetDeviceBuffer(outputs_[0]->Data()));
  cl_int4 input_size = {inputs_[0]->shape()[0], inputs_[0]->shape()[1], inputs_[0]->shape()[2], inputs_[0]->shape()[3]};
  int arg_idx = 0;
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, *input);
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, *output);
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, input_size);

  // run opengl kernel
  ocl_runtime->RunKernel(kernel_, global, local, nullptr);

  return 0;
}

kernel::LiteKernel *OpenCLSoftMaxKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                               const std::vector<lite::tensor::Tensor *> &outputs,
                                               OpParameter *opParameter, const lite::Context *ctx,
                                               const kernel::KernelKey &desc) {
  auto *kernel = new SoftmaxOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (inputs[0]->shape()[0] > 1) {
    MS_LOG(ERROR) << "Init `Softmax` kernel failed: Unsupported multi-batch.";
  }
  auto ret = kernel->Init();
  if (0 != ret) {
    MS_LOG(ERROR) << "Init `Softmax` kernel failed!";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_SoftMax, OpenCLSoftMaxKernelCreator)
}  // namespace kernel
}  // namespace mindspore

