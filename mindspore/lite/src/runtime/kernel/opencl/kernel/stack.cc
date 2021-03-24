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

#include "src/runtime/kernel/opencl/kernel/stack.h"
#include <cstring>
#include <string>
#include <algorithm>
#include <set>
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/kernel/opencl/cl/stack.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::opencl::ImageSize;
using mindspore::schema::PrimitiveType_Stack;

namespace mindspore::kernel {

int StackOpenCLKernel::RunAxis0() {
  auto allocator_ = ocl_runtime_->GetAllocator();
  ImageSize img_size;
  auto dst_data = out_tensors_[0]->data_c();
  auto dst_origin = cl::array<cl::size_type, 3U>{0, 0, 0};
  cl::Image2D *out_image = reinterpret_cast<cl::Image2D *>(allocator_->GetImage(dst_data));
  for (int i = 0; i < in_tensors_.size(); i++) {
    auto src_data = in_tensors_[i]->data_c();
    allocator_->GetImageSize(src_data, &img_size);
    auto src_origin = cl::array<cl::size_type, 3U>{0, 0, 0};
    auto region = cl::array<cl::size_type, 3U>{img_size.width, img_size.height, 1};
    cl::Image2D *input_image = reinterpret_cast<cl::Image2D *>(allocator_->GetImage(src_data));
    ocl_runtime_->GetDefaultCommandQueue()->enqueueCopyImage(*input_image, *out_image, src_origin, dst_origin, region);
    dst_origin[1] += region[1];
  }
  return RET_OK;
}

int StackOpenCLKernel::ReSize() { return RET_OK; }

void StackGetWorkGroup(const std::vector<size_t> &global, std::vector<size_t> *local, int max_size) {
  const int max_divider = 8;
  const int max_x = 4, max_y = 8;
  int x = std::min(GetMaxDivisorStrategy1(global[0], max_divider), max_x);
  int yz = max_size / x;
  int y = std::min(std::min(GetMaxDivisorStrategy1(global[1], max_divider), yz), max_y);
  int z = std::min(yz / y, static_cast<int>(UP_DIV(global[2], 2)));

  local->clear();
  local->push_back(x);
  local->push_back(y);
  local->push_back(z);
}

int StackOpenCLKernel::CheckSpecs() {
  auto param = reinterpret_cast<StackParameter *>(this->op_parameter_);
  axis_ = param->axis_;
  if (in_tensors_.size() != 2 && out_tensors_.size() != 1) {
    MS_LOG(ERROR) << " only support input size = 2 and output size = 1";
    return RET_ERROR;
  }
  for (auto &tensor : in_tensors_) {
    if (tensor->data_type() != kNumberTypeFloat32 && tensor->data_type() != kNumberTypeFloat16) {
      MS_LOG(ERROR) << " only support fp32/fp16 input";
      return RET_ERROR;
    }
  }
  for (auto &tensor : out_tensors_) {
    if (tensor->data_type() != kNumberTypeFloat32 && tensor->data_type() != kNumberTypeFloat16) {
      MS_LOG(ERROR) << " only support fp32/fp16 output";
      return RET_ERROR;
    }
  }
  if (in_tensors_[0]->shape().size() > 4 || in_tensors_[0]->shape().size() <= 0) {
    MS_LOG(ERROR) << " only support 0<dim<=4";
    return RET_ERROR;
  }
  axis_ = axis_ < 0 ? axis_ + in_tensors_[0]->shape().size() : axis_;
  if (axis_ > 3) {
    MS_LOG(ERROR) << " only support  axis <= 3 ";
    return RET_ERROR;
  }
  if (axis_ > in_tensors_[0]->shape().size()) {
    MS_LOG(ERROR) << " stack  axis must been <= in_tensors_[0]->shape().size() ";
    return RET_ERROR;
  }
  return RET_OK;
}

void StackOpenCLKernel::SetConstArgs() {
  int arg_cn = in_tensors_.size() + 1;
  cl_int4 inshape_tmp = {}, outshape_tmp = {};
  for (int i = 0; i < in_tensors_[0]->shape().size(); ++i) {
    inshape_tmp.s[i] = in_tensors_[0]->shape()[i];
  }
  Broadcast2GpuShape(in_shape_.s, inshape_tmp.s, in_tensors_[0]->shape().size(), 1);
  for (int i = 0; i < out_tensors_[0]->shape().size(); ++i) {
    outshape_tmp.s[i] = out_tensors_[0]->shape()[i];
  }
  Broadcast2GpuShape(out_shape_.s, outshape_tmp.s, out_tensors_[0]->shape().size(), 1);
  in_shape_.s[3] = UP_DIV(in_shape_.s[3], C4NUM);
  out_shape_.s[3] = UP_DIV(out_shape_.s[3], C4NUM);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_shape_);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_shape_);
  if (buffer_button_) {
    GpuTensorInfo img_info_out(out_tensors_[0]);
    GpuTensorInfo img_info_in(in_tensors_[0]);
    size_t dtype = enable_fp16_ ? sizeof(cl_half) : sizeof(cl_float);
    stride_w_out = img_info_out.RowPitch() / dtype;
    stride_w_in = img_info_in.RowPitch() / dtype;
    cl_int2 stride_w = {stride_w_out, stride_w_in};
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, stride_w);
  }
}

void StackOpenCLKernel::SetGlobalLocal() {
  if (((in_tensors_[0]->shape().size() == 2 || in_tensors_[0]->shape().size() == 3) && axis_ == 1) ||
      (in_tensors_[0]->shape().size() == 3 && axis_ == 2)) {
    OH_ = out_shape_.s[0] * out_shape_.s[1];
    OW_ = out_shape_.s[2];
    OC_ = out_shape_.s[3];
  } else if (in_tensors_[0]->shape().size() == 1) {
    OH_ = UP_DIV(out_shape_.s[0], C4NUM);
    OW_ = out_shape_.s[3];
  } else {
    OH_ = out_shape_.s[0];
    OW_ = out_shape_.s[1];
  }
  const std::vector<size_t> &max_global = ocl_runtime_->GetWorkItemSize();
  std::vector<size_t> local = {1, 1, 1};
  std::vector<size_t> global = {OH_, OW_, OC_};
  StackGetWorkGroup(global, &local, max_global[0]);
  OpenCLKernel::AlignGlobalLocal(global, local);
}

int StackOpenCLKernel::Prepare() {
  enable_fp16_ = ocl_runtime_->GetFp16Enable();
  if (axis_ == 0) {
    return RET_OK;
  }
  if (in_tensors_[0]->shape().size() == 1 && axis_ == 1) {
    axis_ += 2;
  } else if (in_tensors_[0]->shape().size() == axis_) {
    buffer_button_ = true;  // boundary stack judge
  }
  std::string kernel_name = "stack_";
  if (!buffer_button_) {
    kernel_name += std::to_string(in_tensors_.size()) + "input_" + std::to_string(axis_) + "axis_" +
                   std::to_string(in_tensors_[0]->shape().size()) + "inshape";
  } else {
    kernel_name += std::to_string(in_tensors_.size()) + "input_" + "boundary";
  }

  MS_LOG(DEBUG) << "kernel_name=: " << kernel_name;
  std::string source = stack_source;
  std::string program_name = "stack";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);
  SetConstArgs();
  SetGlobalLocal();

  return RET_OK;
}

int StackOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  if (axis_ == 0) {
    return RunAxis0();
  }
  int arg_cn = 0;
  if (buffer_button_) {
    for (int i = 0; i < in_tensors_.size(); ++i) {
      ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_tensors_[i]->data_c(), lite::opencl::MemType::BUF);
    }
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_tensors_[0]->data_c(), lite::opencl::MemType::BUF);
  } else {
    for (int i = 0; i < in_tensors_.size(); ++i) {
      ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_tensors_[i]->data_c());
    }
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_tensors_[0]->data_c());
  }
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return RET_OK;
}
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Stack, OpenCLKernelCreator<StackOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Stack, OpenCLKernelCreator<StackOpenCLKernel>);

}  // namespace mindspore::kernel
