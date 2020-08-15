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

#include "src/runtime/kernel/opencl/kernel/softmax.h"
#include <string>
#include <set>
#include "include/errorcode.h"
#include "src/kernel_registry.h"
#include "src/runtime/opencl/opencl_runtime.h"
#include "src/runtime/kernel/opencl/utils.h"
#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/fp32/softmax.cl.inc"
#include "src/runtime/kernel/opencl/cl/fp32/softmax1x1.cl.inc"
#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_SoftMax;

namespace mindspore::kernel {

std::vector<float> SoftmaxOpenCLKernel::GetMaskForLastChannel(int channels) {
  std::vector<float> mask{4, 0.0f};
  const int reminder = channels % 4 == 0 ? 4 : channels % 4;
  for (int i = 0; i < reminder; ++i) {
    mask[i] = 1.0f;
  }
  return mask;
}

int SoftmaxOpenCLKernel::InitGlobalSize() {
  const size_t global_x = out_tensors_[0]->Height();
  const size_t global_y = out_tensors_[0]->Width();
  const size_t global_z = 1;
  global_size_ = {global_x, global_y, global_z};
  return lite::RET_OK;
}

int SoftmaxOpenCLKernel::SetWorkGroupSize() {
  // set work group size
  InitGlobalSize();
  int max_work_group_size = runtime_->GetKernelMaxWorkGroupSize(kernel_(), (*runtime_->Device())());
  local_size_ = GetCommonLocalSize(global_size_, max_work_group_size);
  global_size_ = GetCommonGlobalSize(local_size_, global_size_);
  return lite::RET_OK;
}

int SoftmaxOpenCLKernel::SetWorkGroupSize1x1() {
  local_size_ = {32, 1, 1};
  global_size_ = {32, 1, 1};
  return lite::RET_OK;
}

int SoftmaxOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t im_dst_x, im_dst_y;
  if (onexone_flag_) {
    im_dst_x = UP_DIV(in_tensors_[0]->shape()[1], C4NUM);
    im_dst_y = 1;
  } else {
    size_t CO4 = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
    im_dst_x = out_tensors_[0]->Width() * CO4;
    im_dst_y = out_tensors_[0]->Height();
  }
#ifdef ENABLE_FP16
  size_t img_dtype = CL_HALF_FLOAT;
#else
  size_t img_dtype = CL_FLOAT;
#endif
  img_size->clear();
  std::vector<size_t> vec{im_dst_x, im_dst_y, img_dtype};
  *img_size = vec;
  return RET_OK;
}

int SoftmaxOpenCLKernel::Init() {
  std::string kernel_name = "SoftMax";
  std::string program_name = "SoftMax";
  std::string source = softmax_source_fp32;
  runtime_ = lite::opencl::OpenCLRuntime::GetInstance();

  if (in_tensors_[0]->shape().size() == 4 && parameter_->axis_ == 3) {
    // support 4d tensor
    onexone_flag_ = false;
  } else if (in_tensors_[0]->shape().size() == 2 && parameter_->axis_ == 1) {
    // support 2d tensor
    kernel_name += "1x1";
    program_name += "1x1";
    source = softmax1x1_source_fp32;
    onexone_flag_ = true;
  } else {
    MS_LOG(EXCEPTION) << "Init `Softmax` kernel failed: Unsupported axis: " << parameter_->axis_;
  }
#ifdef PROGRAM_WITH_IL
  runtime_->CreateKernelFromIL(kernel_(), kernel_name);
#else
  if (mem_type_ == MEM_TYPE::BUF) {
    kernel_name += "_BUF";
    program_name += "_BUF";
  } else {
    kernel_name += "_IMG";
    program_name += "_IMG";
  }
  std::set<std::string> build_options;
  runtime_->LoadSource(program_name, source);
  out_tensors_[0]->SetFormat(schema::Format_NHWC4);
  runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return lite::RET_OK;
}

int SoftmaxOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";

  int arg_idx = 0;
  if (onexone_flag_) {
    int channel_size = in_tensors_[0]->shape()[1];
    int slices = UP_DIV(channel_size, C4NUM);
    cl_int slices_x32 = UP_DIV(slices, 32);
    auto mask_ = GetMaskForLastChannel(channel_size);
    cl_float4 mask = {mask_[0], mask_[1], mask_[2], mask_[3]};

    runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->Data());
    runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->Data());
    runtime_->SetKernelArg(kernel_, arg_idx++, mask);
    runtime_->SetKernelArg(kernel_, arg_idx++, slices);
    runtime_->SetKernelArg(kernel_, arg_idx, slices_x32);
    SetWorkGroupSize1x1();
  } else {
    int slices = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
    cl_int4 input_shape = {in_tensors_[0]->Height(), in_tensors_[0]->Width(), in_tensors_[0]->Channel(), slices};

    runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->Data());
    runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->Data());
    runtime_->SetKernelArg(kernel_, arg_idx, input_shape);
    SetWorkGroupSize();
  }

  // run opengl kernel
  runtime_->RunKernel(kernel_, global_size_, local_size_, nullptr);
  return lite::RET_OK;
}

kernel::LiteKernel *OpenCLSoftMaxKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                               const std::vector<lite::tensor::Tensor *> &outputs,
                                               OpParameter *opParameter, const lite::Context *ctx,
                                               const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  auto *kernel = new (std::nothrow)SoftmaxOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel " << opParameter->name_ << "is nullptr.";
    return nullptr;
  }
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
}  // namespace mindspore::kernel
