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
#include <cstring>
#include <string>
#include <algorithm>
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/kernel/opencl/cl/concat.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::opencl::ImageSize;
using mindspore::schema::PrimitiveType_Concat;

namespace mindspore::kernel {

int ConcatOpenCLKernel::RunAxis0() {
  auto allocator_ = ocl_runtime_->GetAllocator();
  ImageSize img_size;
  auto dst_data = out_tensors_[0]->data_c();
  auto dst_origin = cl::array<cl::size_type, 3U>{0, 0, 0};
  auto *out_image = reinterpret_cast<cl::Image2D *>(allocator_->GetImage(dst_data));
  for (int i = 0; i < in_tensors_.size(); i++) {
    auto src_data = weight_ptrs_.at(i) == nullptr ? in_tensors_[i]->data_c() : weight_ptrs_.at(i);
    allocator_->GetImageSize(src_data, &img_size);
    auto src_origin = cl::array<cl::size_type, 3U>{0, 0, 0};
    auto region = cl::array<cl::size_type, 3U>{img_size.width, img_size.height, 1};
    auto *input_image = reinterpret_cast<cl::Image2D *>(allocator_->GetImage(src_data));
    ocl_runtime_->GetDefaultCommandQueue()->enqueueCopyImage(*input_image, *out_image, src_origin, dst_origin, region);
    dst_origin[1] += region[1];
  }
  return RET_OK;
}

void ConcatGetWorkGroup(const std::vector<size_t> &global, std::vector<size_t> *local, int max_size) {
  const int max_divider = 8;
  const int max_x = 2, max_y = 8;
  int x = std::min(GetMaxDivisorStrategy1(global[0], max_divider), max_x);
  int yz = max_size / x;
  int y = std::min(std::min(GetMaxDivisorStrategy1(global[1], max_divider), yz), max_y);
  int z = std::min(yz / y, static_cast<int>(UP_DIV(global[2], 2)));

  local->clear();
  local->push_back(x);
  local->push_back(y);
  local->push_back(z);
}

int ConcatOpenCLKernel::CheckSpecs() {
  if ((in_tensors_.size() < 2 || in_tensors_.size() > 6) || out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  auto param = reinterpret_cast<ConcatParameter *>(this->op_parameter_);
  auto out_tensors_shape_size = out_tensors_[0]->shape().size();
  MS_LOG(DEBUG) << " concat at axis=:  " << param->axis_;
  if (out_tensors_shape_size > 4) {
    MS_LOG(ERROR) << " GPU Unsupported shape.size > 4 ";
    return RET_ERROR;
  }
  for (auto &in_tensor : in_tensors_) {
    auto in_tensors_shape_size = in_tensor->shape().size();
    if (in_tensors_shape_size > 4) {
      MS_LOG(ERROR) << " GPU Unsupported in_tensor shape.size > 4 ";
      return RET_ERROR;
    }
  }
  axis_ = param->axis_;
  if (axis_ < 0) {
    axis_ += in_tensors_.front()->shape().size();
  }
  if (axis_ < 0 || axis_ > 3) {
    MS_LOG(ERROR) << " only support axis >= 0 and axis <= 3 ";
    return RET_ERROR;
  }
  if (out_tensors_shape_size < 4 && Type() == PrimitiveType_Concat && axis_ != 0) {
    if (out_tensors_shape_size == 2) {
      axis_ = axis_ + 2;
    } else if (out_tensors_shape_size == 3) {
      axis_ = axis_ + 1;
    } else {
      MS_LOG(ERROR) << " Unsupported axis =:  " << axis_ << "  shape().size()=:  " << out_tensors_shape_size;
      return RET_ERROR;
    }
  }
  if (in_tensors_.size() < 2 || in_tensors_.size() > 6) {
    MS_LOG(ERROR) << "unsupported input size :" << in_tensors_.size();
    return RET_ERROR;
  }
  return RET_OK;
}

void ConcatOpenCLKernel::SetConstArgs() {
  GpuTensorInfo img_info(out_tensors_[0]);
  size_t dtype = ocl_runtime_->GetFp16Enable() ? sizeof(cl_half) : sizeof(cl_float);
  stride_w = img_info.RowPitch() / dtype;
  cl_int4 output_shape_ = {};
  for (int i = 0; i < out_tensors_[0]->shape().size(); ++i) {
    output_shape_.s[i] = out_tensors_[0]->shape()[i];
  }
  Broadcast2GpuShape(out_shape_.s, output_shape_.s, out_tensors_[0]->shape().size(), 1);
  int arg_cn = in_tensors_.size() + 1;
  if (axis_ == 3 && !Align_) {
    for (auto &in_tensor : in_tensors_) {
      cl_int4 temp = {};
      for (int j = 0; j < in_tensor->shape().size(); ++j) {
        temp.s[j] = in_tensor->shape()[j];
      }
      Broadcast2GpuShape(in_shape_.s, temp.s, in_tensor->shape().size(), 1);
      ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_shape_);
    }
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, stride_w);
  } else {
    for (auto &in_tensor : in_tensors_) {
      cl_int4 temp = {};
      for (int j = 0; j < in_tensor->shape().size(); ++j) {
        temp.s[j] = in_tensor->shape()[j];
      }
      Broadcast2GpuShape(in_shape_.s, temp.s, in_tensor->shape().size(), 1);
      in_shape_.s[3] = UP_DIV(in_shape_.s[3], C4NUM);
      ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_shape_);
    }
  }
  out_shape_.s[3] = UP_DIV(out_shape_.s[3], C4NUM);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_shape_);
}

void ConcatOpenCLKernel::SetGlobalLocal() {
  const std::vector<size_t> &max_global = ocl_runtime_->GetWorkItemSize();
  if (axis_ == 3 && !Align_) {
    OH = out_shape_.s[0] * out_shape_.s[1];
    OW = out_shape_.s[2];
    global_size_ = {OH, OW, 1};
    local_size_ = {1, 1, 1};
  } else {
    OH = out_shape_.s[0] * out_shape_.s[1];
    OW = out_shape_.s[2];
    OC = out_shape_.s[3];
    global_size_ = {OH, OW, OC};
    local_size_ = {1, 1, 1};
  }
  ConcatGetWorkGroup(global_size_, &local_size_, max_global[0]);
  OpenCLKernel::AlignGlobalLocal(global_size_, local_size_);
}

int ConcatOpenCLKernel::ConvertWeightToTensor() {
  auto allocator = ocl_runtime_->GetAllocator();
  bool fp16_enable = ocl_runtime_->GetFp16Enable();
  for (auto in_tensor : in_tensors_) {
    auto in_shape = GpuTensorInfo(in_tensor);
    if (in_tensor->IsConst()) {
      std::vector<char> weight(in_shape.Image2DSize, 0);
      bool src_is_fp16 = in_tensor->data_type() == kNumberTypeFloat16;
      PackNHWCToNHWC4(in_tensor->data_c(), weight.data(), src_is_fp16,
                      fp16_enable && in_tensor->data_type() != kNumberTypeInt32, in_shape);
      size_t dtype = fp16_enable && in_tensor->data_type() != kNumberTypeInt32 ? CL_HALF_FLOAT : CL_FLOAT;
      ImageSize img_size{in_shape.width, in_shape.height, dtype};
      auto weight_ptr_ = allocator->Malloc(img_size, weight.data());
      weight_ptrs_.push_back(weight_ptr_);
    } else {
      weight_ptrs_.push_back(nullptr);
    }
  }
  return RET_OK;
}

int ConcatOpenCLKernel::Prepare() {
  ConvertWeightToTensor();
  if (axis_ == 0) {
    if (std::any_of(in_tensors_.begin(), in_tensors_.end(), [](lite::Tensor *t) { return t->shape().size() != 1; })) {
      return RET_OK;
    }
    axis_ = 3;
  }
  for (auto const &in_tensor : in_tensors_) {
    if (in_tensor->shape().back() % C4NUM != 0) {
      Align_ = false;
    }
  }

  std::string kernel_name = "Concat";
  if (axis_ == 3 && !Align_) {
    kernel_name += "Input" + std::to_string(in_tensors_.size()) + "UnAlign";
  } else {
    kernel_name += std::to_string(in_tensors_.size()) + "inputaxis" + std::to_string(axis_);
  }

  kernel_name += "_NHWC4";
  MS_LOG(DEBUG) << "kernel_name=: " << kernel_name;
  std::string source = concat_source;
  std::string program_name = "Concat";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, {}, out_tensors_[0]->data_type());
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  SetConstArgs();
  SetGlobalLocal();
  return RET_OK;
}

int ConcatOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  if (axis_ == 0) {
    return RunAxis0();
  }
  int arg_cn = 0;
  for (int i = 0; i < in_tensors_.size(); ++i) {
    auto input_ptr = weight_ptrs_.at(i) == nullptr ? in_tensors_[i]->data_c() : weight_ptrs_.at(i);
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, input_ptr);
  }
  if (axis_ == 3 && !Align_) {
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_tensors_[0]->data_c(), lite::opencl::MemType::BUF);
  } else {
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_tensors_[0]->data_c());
  }
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Concat, OpenCLKernelCreator<ConcatOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Concat, OpenCLKernelCreator<ConcatOpenCLKernel>)
}  // namespace mindspore::kernel
