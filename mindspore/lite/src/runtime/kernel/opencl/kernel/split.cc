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

#include "src/runtime/kernel/opencl/kernel/split.h"
#include <cstring>
#include <string>
#include <algorithm>
#include <set>
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/cl/split.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::opencl::ImageSize;
using mindspore::schema::PrimitiveType_Split;

namespace mindspore::kernel {

int SplitOpenCLKernel::RunAxis0() {
  auto allocator_ = ocl_runtime_->GetAllocator();
  auto src_data = in_tensors_[0]->data_c();
  cl::Image2D *in_image = reinterpret_cast<cl::Image2D *>(allocator_->GetImage(src_data));
  if (in_image == nullptr) {
    MS_LOG(ERROR) << "RunAxis0 in_image can not be nullptr";
    return RET_ERROR;
  }
  auto src_area = cl::array<cl::size_type, 3U>{0, 0, 0};
  for (int i = 0; i < out_tensors_.size(); i++) {
    auto dst_data = out_tensors_[i]->data_c();
    ImageSize img_size;
    allocator_->GetImageSize(dst_data, &img_size);
    auto dst_area = cl::array<cl::size_type, 3U>{0, 0, 0};
    auto region = cl::array<cl::size_type, 3U>{img_size.width, img_size.height, 1};
    cl::Image2D *out_image = reinterpret_cast<cl::Image2D *>(allocator_->GetImage(dst_data));
    if (out_image == nullptr) {
      MS_LOG(ERROR) << "RunAxis0 out_image can not be nullptr";
      return RET_ERROR;
    }
    ocl_runtime_->GetDefaultCommandQueue()->enqueueCopyImage(*in_image, *out_image, src_area, dst_area, region);
    src_area[1] += region[1];
  }
  return RET_OK;
}

int SplitOpenCLKernel::CheckSpecs() {
  auto param = reinterpret_cast<SplitParameter *>(this->op_parameter_);
  if ((out_tensors_.size() != 2 || (out_tensors_.size() != 3 && param->split_dim_ == 0)) && in_tensors_.size() != 1) {
    MS_LOG(ERROR) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  if (in_tensors_.at(0)->IsConst()) {
    MS_LOG(ERROR) << "in_tensors_ must be tensor";
    return RET_ERROR;
  }
  for (auto &out_tensor : out_tensors_) {
    if (out_tensor->IsConst()) {
      MS_LOG(ERROR) << "out_tensor must be tensor";
      return RET_ERROR;
    }
  }

  if (param->num_split_ != 2 && (param->num_split_ != 3 && param->split_dim_ == 0)) {
    MS_LOG(ERROR) << "num_split_ only supported 2 or (3 && split_dim_ = 0) yet";
    return RET_ERROR;
  }
  if (param->split_dim_ < 0 || param->split_dim_ > 3) {
    MS_LOG(ERROR) << "split_dim_ must between 0~3";
    return RET_ERROR;
  }
  if (param->split_sizes_ == nullptr) {
    MS_LOG(ERROR) << "split_sizes_ can not nullptr";
    return RET_ERROR;
  }
  return RET_OK;
}

void SplitOpenCLKernel::AlignSplitSizes(SplitParameter *param, const std::vector<int> &in_shape) {
  auto allocator = ocl_runtime_->GetAllocator();
  int shape_dim = in_shape.at(param->split_dim_);
  if (num_split_ == 1) {
    size_t num_split = UP_DIV(shape_dim, param->split_sizes_[0]);
    split_sizes_ = reinterpret_cast<int *>(allocator->Malloc(num_split * sizeof(int)));
    for (int i = 0; i < num_split - 1; ++i) {
      split_sizes_[i] = (i + 1) * param->split_sizes_[0];
    }
  } else {
    int sum = 0;
    split_sizes_ = reinterpret_cast<int *>(allocator->Malloc(num_split_ * sizeof(int)));
    for (int i = 0; i < num_split_ - 1; ++i) {
      sum += param->split_sizes_[i];
      split_sizes_[i] = sum;
    }
  }
}

int SplitOpenCLKernel::Prepare() {
  auto param = reinterpret_cast<SplitParameter *>(this->op_parameter_);
  auto in_shape = in_tensors_.at(0)->shape();
  int increment_dim = C4NUM - in_shape.size();
  split_dim_ = param->split_dim_ == 0 ? param->split_dim_ : param->split_dim_ + increment_dim;
  num_split_ = param->num_split_;
  if (split_dim_ == 0) {
    return RET_OK;
  }
  for (int i = 0; i < out_tensors_.size(); ++i) {
    int length = out_tensors_[0]->shape().size();
    if (split_dim_ == 3) {
      if (out_tensors_[i]->shape()[length - 1] % C4NUM != 0) {
        Align_ = false;
      }
    }
  }
  AlignSplitSizes(param, in_shape);
  std::string kernel_name = "split_out";
  kernel_name += num_split_ == 1 ? std::to_string(out_tensors().size()) : std::to_string(num_split_);
  kernel_name += "_axis" + std::to_string(split_dim_);
  if (!Align_) {
    kernel_name += "_unalign";
  }
  MS_LOG(DEBUG) << "kernel_name=: " << kernel_name;
  std::string source = split_source;
  std::string program_name = "split";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  SetConstArgs();
  SetGlobalLocal();
  return RET_OK;
}

void SplitOpenCLKernel::SetConstArgs() {
  int arg_cn = out_tensors_.size() + 2;
  cl_int4 shape = {};
  for (int i = 0; i < in_tensors_[0]->shape().size(); ++i) {
    shape.s[i] = in_tensors_[0]->shape()[i];
  }
  Broadcast2GpuShape(in_shape_.s, shape.s, out_tensors_[0]->shape().size(), 1);
  if (Align_) {
    in_shape_.s[3] = UP_DIV(in_shape_.s[3], C4NUM);
  }
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_shape_);

  for (int i = 0; i < out_tensors_.size(); ++i) {
    cl_int4 temp = {};
    for (int j = 0; j < out_tensors_[i]->shape().size(); ++j) {
      temp.s[j] = out_tensors_[i]->shape()[j];
    }
    Broadcast2GpuShape(out_shape_.s, temp.s, out_tensors_[i]->shape().size(), 1);
    if (Align_) {
      out_shape_.s[3] = UP_DIV(out_shape_.s[3], C4NUM);
    }
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_shape_);
  }
  GpuTensorInfo img_info(in_tensors_.at(0));
  size_t dtype = enable_fp16_ ? sizeof(cl_half) : sizeof(cl_float);
  stride_w = img_info.RowPitch() / dtype;
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, stride_w);
  return;
}

void SplitOpenCLKernel::SetGlobalLocal() {
  OH = in_shape_.s[0] * in_shape_.s[1];
  OW = in_shape_.s[2];
  if (Align_) {
    OC = in_shape_.s[3];
  }
  global_size_ = {OH, OW, OC};
  local_size_ = {1, 1, 1};
  OpenCLKernel::AlignGlobalLocal(global_size_, local_size_);
  return;
}

int SplitOpenCLKernel::Run() {
  if (split_dim_ == 0) {
    RunAxis0();
    return RET_OK;
  }
  int arg_cn = 0;
  if (Align_) {
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_tensors_.at(0)->data_c());
  } else {
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_tensors_.at(0)->data_c(), lite::opencl::MemType::BUF);
  }
  for (int i = 0; i < out_tensors_.size(); ++i) {
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_tensors_.at(i)->data_c());
  }
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, split_sizes_, lite::opencl::MemType::BUF);
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Split, OpenCLKernelCreator<SplitOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Split, OpenCLKernelCreator<SplitOpenCLKernel>)
}  // namespace mindspore::kernel
