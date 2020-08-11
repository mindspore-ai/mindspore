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
#include <cstring>
#include <string>
#include <algorithm>
#include <set>
#include "src/kernel_registry.h"
#include "src/runtime/opencl/opencl_runtime.h"
#include "src/runtime/kernel/opencl/kernel/concat.h"
#include "src/runtime/kernel/opencl/cl/fp32/concat.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Concat;

namespace mindspore::kernel {

int ConcatOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t CO4 = UP_DIV(outputs_[0]->Channel(), C4NUM);
  size_t im_dst_x, im_dst_y;
  if (inputs_[0]->GetFormat() == schema::Format_NHWC4) {
    im_dst_x = outputs_[0]->Width() * CO4;
    im_dst_y = outputs_[0]->Height();
  } else {
    im_dst_y = outputs_[0]->Height() * CO4;
    im_dst_x = outputs_[0]->Width();
  }
#ifdef ENABLE_FP16
  size_t img_dtype = CL_HALF_FLOAT;
#else
  size_t img_dtype = CL_FLOAT;
#endif
  img_size->clear();
  std::vector<size_t> vec{im_dst_x, im_dst_y, img_dtype};
  *img_size = vec;
  return 1;
}
int ConcatOpenCLKernel::Init() {
  if (inputs_[0]->shape().size() != 4) {
    MS_LOG(ERROR) << "only support dim=4";
  }

  auto param = reinterpret_cast<ConcatParameter *>(this->opParameter);
  MS_LOG(INFO) << "concat at axis=:  " << param->axis_;
  if (param->axis_ != 0 && param->axis_ != 3) {
    MS_LOG(ERROR) << "only support axis=0 or axis=3";
  }

  if (param->axis_ == 0) {
    return 0;
  }
  if (inputs_.size() == 2) {
    std::set<std::string> build_options;
    std::string source = concat_source_fp32;
    std::string program_name = "Concat";
    std::string kernel_name = "Concat";
    auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
    ocl_runtime->LoadSource(program_name, source);
    ocl_runtime->BuildKernel(kernel_, program_name, kernel_name, build_options);
  }

  if (inputs_.size() == 3) {
    std::set<std::string> build_options;
    std::string source = concat_source_fp32;
    std::string program_name = "Concat3input";
    std::string kernel_name = "Concat3input";
    auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
    ocl_runtime->LoadSource(program_name, source);
    ocl_runtime->BuildKernel(kernel_, program_name, kernel_name, build_options);
  }

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

  memcpy(outputs_[0]->Data(), inputs_[0]->Data(), inputs_[0]->Size());
  memcpy(reinterpret_cast<char *>(outputs_[0]->Data()) + inputs_[0]->Size(), inputs_[1]->Data(), inputs_[1]->Size());

  for (auto tensors : {&inputs_, &outputs_}) {
    for (auto &tensor : *tensors) {
      auto buffer = static_cast<cl::Buffer *>(allocator_->GetDeviceBuffer(tensor->Data()));
      ocl_runtime->UnmapBuffer(*buffer, tensor->Data());
    }
  }
  return 0;
}
int DivideRoundUp(int n, int div) {
  int q = n / div;
  return n % div == 0 ? q : q + 1;
}

int GetBiggestDividerWithPriority(int number, int max_divider) {
  if (number % 8 == 0 && 8 <= max_divider) {
    return number / 8;
  }
  if (number % 4 == 0 && 4 <= max_divider) {
    return number / 4;
  }
  if (number % 2 == 0 && 2 <= max_divider) {
    return number / 2;
  }
  for (int i = max_divider; i != 0; i--) {
    if (number % i == 0) {
      return i;
    }
  }
  return 1;
}

void ConcatGetWorkGroup(const std::vector<size_t> &global, std::vector<size_t> *local, int max_size) {
  int x = std::min(GetBiggestDividerWithPriority(global[0], 8), 4);
  int yz = max_size / x;
  int y = std::min(std::min(GetBiggestDividerWithPriority(global[1], 8), yz), 8);
  int z = std::min(yz / y, DivideRoundUp(global[2], 2));

  local->clear();
  local->push_back(x);
  local->push_back(y);
  local->push_back(z);
}
int ConcatOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->Name() << " Running!";
  auto param = reinterpret_cast<ConcatParameter *>(this->opParameter);
  if (param->axis_ == 0) {
    return Run_axis0();
  }

  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  auto input0_shape = inputs_[0]->shape();
  auto input1_shape = inputs_[1]->shape();
  auto output_shape = outputs_[0]->shape();

  cl_int2 input0_shape2_ = {DivideRoundUp(input0_shape[3], 4), DivideRoundUp(input1_shape[3], 4)};  // change
  cl_int4 output_shape_ = {output_shape[0], output_shape[1], output_shape[2], DivideRoundUp(output_shape[3], 4)};

  uint32_t OH = output_shape[0] * output_shape[1];  // N*H
  uint32_t OW = output_shape[2];

  std::vector<size_t> local = {1, 1};
  std::vector<size_t> global = {OH, OW};
  //    ConcatGetWorkGroup(global, &local, 512);

  int arg_cn = 0;
  if (inputs_.size() == 2) {
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, outputs_[0]->Data());
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, inputs_[0]->Data());
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, inputs_[1]->Data());
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, input0_shape2_);
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, output_shape_);
  } else if (inputs_.size() == 3) {
    auto input2_shape = inputs_[2]->shape();
    cl_int3 input0_shape3_ = {DivideRoundUp(input0_shape[3], 4), DivideRoundUp(input1_shape[3], 4),
                              DivideRoundUp(input2_shape[3], 4)};
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, outputs_[0]->Data());
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, inputs_[0]->Data());
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, inputs_[1]->Data());
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, inputs_[2]->Data());
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, input0_shape3_);
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, output_shape_);
  }
  ocl_runtime->RunKernel(kernel_, global, local, nullptr);

  return 0;
}  // namespace mindspore::kernel

kernel::LiteKernel *OpenCLConcatKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                              const std::vector<lite::tensor::Tensor *> &outputs,
                                              OpParameter *opParameter, const lite::Context *ctx,
                                              const kernel::KernelKey &desc, const lite::Primitive *primitive) {
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
