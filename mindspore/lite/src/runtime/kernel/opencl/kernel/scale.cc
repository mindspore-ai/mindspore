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

#include "src/runtime/kernel/opencl/kernel/scale.h"
#include <set>
#include <vector>
#include <string>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "nnacl/fp32/common_func_fp32.h"
#include "src/runtime/kernel/opencl/utils.h"
#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/scale.cl.inc"
#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::opencl::ImageSize;
using mindspore::lite::opencl::MemType;
using mindspore::schema::PrimitiveType_ScaleFusion;

namespace mindspore::kernel {

int ScaleOpenCLKernel::CheckSpecs() {
  auto *param = reinterpret_cast<const ScaleParameter *>(op_parameter_);
  if (param->activation_type_ != ActType_No && param->activation_type_ != ActType_Relu &&
      param->activation_type_ != ActType_Relu6) {
    return RET_ERROR;
  }
  return RET_OK;
}

ScaleOpenCLKernel::~ScaleOpenCLKernel() {
  auto allocator = ocl_runtime_->GetAllocator();
  if (scale_ptr_ != nullptr) {
    allocator->Free(scale_ptr_);
    scale_ptr_ = nullptr;
  }
  if (offset_ptr_ != nullptr) {
    allocator->Free(offset_ptr_);
    offset_ptr_ = nullptr;
  }
}

void ScaleOpenCLKernel::Image2dGetWorkGroupSize() {
  local_size_ = {16, 16};
  auto image2d_info = GpuTensorInfo(out_tensors_[0]);
  global_size_ = {image2d_info.width, image2d_info.height};
  OpenCLKernel::AlignGlobalLocal(global_size_, local_size_);
}

int ScaleOpenCLKernel::InitWeights() {
  auto *in_tensor = in_tensors_[0];
  auto *scale_tensor = in_tensors_[1];
  auto *offset_tensor = in_tensors_[2];
  auto scale_dtype = scale_tensor->data_type();
  if (!weight_vector_flag_ || !scale_tensor->IsConst()) {
    return RET_OK;
  }
  auto allocator = ocl_runtime_->GetAllocator();
  auto fp16_enable = ocl_runtime_->GetFp16Enable();
  ImageSize img_size;
  GetImageSize(0, &img_size);
  img_size.dtype = scale_dtype == kNumberTypeFloat16 ? CL_HALF_FLOAT : CL_FLOAT;

  if (broadcast_flag_) {
    img_size.height = 1;
    img_size.width = UP_DIV(scale_tensor->shape()[0], C4NUM);
    scale_ptr_ = allocator->Malloc(img_size, scale_tensor->data_c());
    offset_ptr_ = allocator->Malloc(img_size, offset_tensor->data_c());
    return RET_OK;
  }

  if (in_tensor->format() == scale_tensor->format()) {
    if (in_tensor->data_type() == scale_tensor->data_type()) {
      scale_ptr_ = allocator->Malloc(img_size, scale_tensor->data_c());
      offset_ptr_ = allocator->Malloc(img_size, offset_tensor->data_c());
    } else {
      MS_LOG(ERROR) << "Unsupported data type transpose from " << scale_tensor->data_type() << "to "
                    << in_tensor->data_type();
      return RET_ERROR;
    }
  } else if (in_tensor->format() == schema::Format_NHWC && scale_tensor->format() == schema::Format_NHWC) {
    if (scale_dtype == kNumberTypeFloat32 || scale_dtype == kNumberTypeFloat16) {
      auto image2d_info = GpuTensorInfo(scale_tensor);
      int pack_weight_size = image2d_info.ElementsC4Num;
      std::vector<char> scale(pack_weight_size, 0);
      std::vector<char> offset(pack_weight_size, 0);
      bool src_is_fp16 = scale_dtype == kNumberTypeFloat16;
      PackNHWCToNHWC4(scale_tensor->data_c(), scale.data(), src_is_fp16, fp16_enable, image2d_info);
      PackNHWCToNHWC4(offset_tensor->data_c(), offset.data(), src_is_fp16, fp16_enable, image2d_info);
      scale_ptr_ = allocator->Malloc(img_size, scale.data());
      offset_ptr_ = allocator->Malloc(img_size, offset.data());
    } else {
      MS_LOG(ERROR) << "Unsupported data type transpose from " << scale_tensor->data_type() << "to "
                    << in_tensor->data_type();
      return RET_ERROR;
    }
  } else {
    MS_LOG(ERROR) << "Unsupported format transpose from " << scale_tensor->format() << "to " << in_tensor->format();
    return RET_ERROR;
  }
  return RET_OK;
}

int ScaleOpenCLKernel::Prepare() {
  std::string kernel_name;
  auto *scale_param = reinterpret_cast<const ScaleParameter *>(op_parameter_);
  auto in_tensor = in_tensors_.at(0);
  auto in_shape = in_tensor->shape();
  auto scale_tensor = in_tensors_.at(1);
  auto scale_shape = scale_tensor->shape();
  axis_ = scale_param->axis_;
  if (axis_ < 0) {
    axis_ += in_shape.size();
  }
  if (scale_shape.size() != in_shape.size()) {
    if (scale_tensor->ElementsNum() == 1) {
      weight_vector_flag_ = false;
      kernel_name = "BoardcastScale";
    } else if (scale_shape.size() == 1) {
      weight_vector_flag_ = true;
      broadcast_flag_ = true;
      if ((in_shape.size() == 4 && axis_ == 3) || (in_shape.size() == 2 && axis_ == 1)) {
        kernel_name = "Scale_C";
      } else if (in_shape.size() == 4 && axis_ == 1) {
        kernel_name = "Scale_H";
        broadcast_H_flag_ = true;
      } else {
        MS_LOG(ERROR) << "unsupported scale axis " << axis_;
        return RET_ERROR;
      }
    } else {
      MS_LOG(ERROR) << "unsupported scale axis " << axis_ << ", in shape " << in_shape << ", scale shape"
                    << scale_shape;
      return RET_ERROR;
    }
  } else {
    weight_vector_flag_ = true;
    kernel_name = "Scale";
  }
  lite::STATUS error_code;
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else
  if (out_mem_type_ == MemType::IMG) {
    kernel_name += "_IMG";
  } else {
    kernel_name += "_BUF";
  }
  std::string program_name = "Scale";
  std::string source = GetActDefines() + scale_source;
  ocl_runtime_->LoadSource(program_name, source);
  error_code = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);
#endif
  if (error_code != RET_OK) {
    return error_code;
  }

  Image2dGetWorkGroupSize();
  InitWeights();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int ScaleOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  auto *param = reinterpret_cast<const ScaleParameter *>(op_parameter_);
  int arg_idx = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->data_c());
  if (weight_vector_flag_) {
    void *scale = scale_ptr_ == nullptr ? in_tensors_[1]->data_c() : scale_ptr_;
    void *offset = offset_ptr_ == nullptr ? in_tensors_[2]->data_c() : offset_ptr_;
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, scale);
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, offset);
  } else {
    if (in_tensors_[1]->data_type() == kNumberTypeFloat32) {
      float scale = static_cast<float *>(in_tensors_[1]->data_c())[0];
      float offset = static_cast<float *>(in_tensors_[2]->data_c())[0];
      ocl_runtime_->SetKernelArg(kernel_, arg_idx++, scale);
      ocl_runtime_->SetKernelArg(kernel_, arg_idx++, offset);
    } else if (in_tensors_[1]->data_type() == kNumberTypeFloat16) {
      float16_t scale = static_cast<float16_t *>(in_tensors_[1]->data_c())[0];
      float16_t offset = static_cast<float16_t *>(in_tensors_[2]->data_c())[0];
      ocl_runtime_->SetKernelArg(kernel_, arg_idx++, static_cast<float>(scale));
      ocl_runtime_->SetKernelArg(kernel_, arg_idx++, static_cast<float>(offset));
    } else {
      MS_LOG(ERROR) << "Unsupported data type " << in_tensors_[1]->data_type();
      return RET_ERROR;
    }
  }
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->data_c());
  cl_int2 output_shape{static_cast<int>(global_size_[0]), static_cast<int>(global_size_[1])};
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, output_shape);
  if (weight_vector_flag_ && broadcast_flag_) {
    if (broadcast_H_flag_) {
      ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[1]->shape()[0]);
    } else {
      ocl_runtime_->SetKernelArg(kernel_, arg_idx++, UP_DIV(in_tensors_[1]->shape()[0], C4NUM));
    }
  }
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, param->activation_type_);
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_ScaleFusion, OpenCLKernelCreator<ScaleOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_ScaleFusion, OpenCLKernelCreator<ScaleOpenCLKernel>)
}  // namespace mindspore::kernel
