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
#include "src/runtime/kernel/opencl/utils.h"
#include "nnacl/softmax_parameter.h"
#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/softmax.cl.inc"
#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Softmax;

namespace mindspore::kernel {

std::vector<float> SoftmaxOpenCLKernel::GetMaskForLastChannel(int channels) {
  std::vector<float> mask{0.0f, 0.0f, 0.0f, 0.0f};
  const int reminder = channels % 4 == 0 ? 4 : channels % 4;
  for (int i = 0; i < reminder; ++i) {
    mask[i] = 1.0f;
  }
  return mask;
}

int SoftmaxOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() != 1 || out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  SoftmaxParameter *parameter = reinterpret_cast<SoftmaxParameter *>(op_parameter_);
  axis_ = parameter->axis_;
  auto in_shape = in_tensors_[0]->shape();
  if (in_shape.size() > 4) {
    MS_LOG(ERROR) << "Init Softmax kernel failed: Unsupported shape size: " << in_shape.size();
    return RET_ERROR;
  }
  if (axis_ < 0) {
    axis_ = in_shape.size() + axis_;
  }
  axis_ += 4 - in_shape.size();
  if (axis_ != 1 && axis_ != 2 && axis_ != 3) {
    MS_LOG(ERROR) << "Init Softmax kernel failed: softmax axis should be H W or C";
    return RET_ERROR;
  }
  return RET_OK;
}

int SoftmaxOpenCLKernel::Prepare() {
  std::string kernel_name = "Softmax";

  out_shape_ = GpuTensorInfo(out_tensors_[0]);
  std::string source = softmax_source;
  if (out_shape_.H == 1 && out_shape_.W == 1 && axis_ == 3) {
    // support 4d tensor
    onexone_flag_ = true;
    kernel_name += "1x1";
  } else {
    onexone_flag_ = false;
    kernel_name += "Axis" + std::to_string(axis_);
  }
  kernel_name += "_NHWC4";
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime->GetKernelFromBinary(kernel_name);
#else
  std::string program_name = "Softmax";
  ocl_runtime_->LoadSource(program_name, source);
  std::vector<std::string> ext_build_opt;
  if (out_tensors_[0]->data_type() == kNumberTypeFloat32) {
    ext_build_opt.push_back("-DOUT_FLT4=convert_float4 -DWRITE_IMAGEOUT=write_imagef");
  } else {
    ext_build_opt.push_back("-DOUT_FLT4=convert_half4 -DWRITE_IMAGEOUT=write_imageh");
  }
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, ext_build_opt);
#endif
  SetConstArgs();
  SetGlobalLocal();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return lite::RET_OK;
}

void SoftmaxOpenCLKernel::SetGlobalLocal() {
  if (onexone_flag_) {
    local_size_ = {32, 1};
    global_size_ = {32, out_shape_.N};
  } else {
    size_t global_x, global_y;
    if (axis_ == 1) {
      global_x = out_shape_.Slice;
      global_y = out_shape_.W;
    } else if (axis_ == 2) {
      global_x = out_shape_.Slice;
      global_y = out_shape_.H;
    } else if (axis_ == 3) {
      global_x = out_shape_.W;
      global_y = out_shape_.H;
    } else {
      global_x = 1;
      global_y = 1;
    }
    global_size_ = {global_x, global_y, out_shape_.N};
    local_size_ = {};
  }
  AlignGlobalLocal(global_size_, local_size_);
}

int SoftmaxOpenCLKernel::Tune() {
  if (onexone_flag_) {
    return RET_OK;
  }
  return OpenCLKernel::Tune();
}

void SoftmaxOpenCLKernel::SetConstArgs() {
  int arg_idx = 2;
  int channel = out_shape_.C;
  int c4 = out_shape_.Slice;
  auto mask_ = GetMaskForLastChannel(channel);
  cl_float4 mask = {mask_[0], mask_[1], mask_[2], mask_[3]};
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, mask);
  cl_int4 input_shape = {static_cast<int>(out_shape_.N), static_cast<int>(out_shape_.H), static_cast<int>(out_shape_.W),
                         c4};
  ocl_runtime_->SetKernelArg(kernel_, arg_idx, input_shape);
}

int SoftmaxOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  int arg_idx = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->data_c());
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return lite::RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Softmax, OpenCLKernelCreator<SoftmaxOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Softmax, OpenCLKernelCreator<SoftmaxOpenCLKernel>)
}  // namespace mindspore::kernel
