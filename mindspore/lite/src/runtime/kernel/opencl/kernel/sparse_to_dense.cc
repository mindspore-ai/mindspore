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

#include "src/runtime/kernel/opencl/kernel/sparse_to_dense.h"
#include <cstring>
#include <string>
#include <algorithm>
#include <set>
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/kernel/opencl/cl/sparse_to_dense.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::opencl::ImageSize;
using mindspore::schema::PrimitiveType_SparseToDense;

namespace mindspore::kernel {

int SparseToDenseOpenCLKernel::InitOutputToDefault() {
  auto allocator_ = ocl_runtime_->GetAllocator();
  ImageSize img_size;
  cl_float4 fill_value = {};
  fill_value.s[0] = fill_value.s[1] = fill_value.s[2] = fill_value.s[3] = default_;
  auto src_data = out_tensors_[0]->data_c();
  allocator_->GetImageSize(src_data, &img_size);
  auto src_origin = cl::array<cl::size_type, 3U>{0, 0, 0};
  auto region = cl::array<cl::size_type, 3U>{img_size.width, img_size.height, 1};
  cl::Image2D *out_image = reinterpret_cast<cl::Image2D *>(allocator_->GetImage(src_data));
  ocl_runtime_->GetDefaultCommandQueue()->enqueueFillImage(*out_image, fill_value, src_origin, region);
  return RET_OK;
}

int SparseToDenseOpenCLKernel::InitWeights() {
  auto allocator = ocl_runtime_->GetAllocator();
  auto weight_tensor = in_tensors_[2];
  size_t size = 1;
  for (int i = 0; i < weight_tensor->shape().size(); ++i) {
    size *= weight_tensor->shape()[i];
  }
  if (weight_scalar_) {
    if (weight_tensor->data_type() == kNumberTypeFloat16) {
      weight_scalar_ = static_cast<float>(*reinterpret_cast<float16_t *>(weight_tensor->data_c()));
    } else {
      weight_scalar_ = *reinterpret_cast<float *>(weight_tensor->data_c());
    }
  } else {
    auto sizeof_FLT = enable_fp16_ ? sizeof(float16_t) : sizeof(float);
    size_t weight_size = UP_ROUND(size, C4NUM) * sizeof_FLT;
    weight_vector_ = allocator->Malloc(weight_size);
    allocator->MapBuffer(weight_vector_, CL_MAP_WRITE, nullptr, true);
    memset(weight_vector_, 0x00, weight_size);
    if (weight_tensor->data_type() == kNumberTypeFloat16) {
      if (enable_fp16_) {
        memcpy(weight_vector_, weight_tensor->data_c(), size * sizeof_FLT);
      } else {
        auto weight_fp32 = reinterpret_cast<float *>(weight_vector_);
        auto origin_bias_fp16 = reinterpret_cast<float16_t *>(weight_tensor->data_c());
        for (int i = 0; i < size; ++i) {
          weight_fp32[i] = static_cast<float>(origin_bias_fp16[i]);
        }
      }
    } else {
      if (enable_fp16_) {
        auto weight_fp16 = reinterpret_cast<float16_t *>(weight_vector_);
        auto origin_bias_fp32 = reinterpret_cast<float *>(weight_tensor->data_c());
        for (int i = 0; i < size; ++i) {
          weight_fp16[i] = static_cast<float16_t>(origin_bias_fp32[i]);
        }
      } else {
        memcpy(weight_vector_, weight_tensor->data_c(), size * sizeof_FLT);
      }
    }
    allocator->UnmapBuffer(weight_vector_);
  }
  return RET_OK;
}

int SparseToDenseOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() < 3 || out_tensors_.at(0)->shape().size() > 4) {
    MS_LOG(ERROR) << " only support out_tensors_ dim <= 4 and in_tensors_.size >= 3";
    return RET_ERROR;
  }
  if (in_tensors_.at(0)->shape().size() > 4 || out_tensors_.at(0)->shape().size() > 4) {
    MS_LOG(ERROR) << "Unsupported inputdim: " << in_tensors_[0]->shape().size() << "outdim"
                  << out_tensors_[0]->shape().size();
    return RET_ERROR;
  }
  if (input_dim_ == 2) {
    if ((in_tensors_[0]->shape()[1] > 4)) {
      MS_LOG(ERROR) << "in_tensors_indices shape[1] must be 1 2 or 3  && input_dim_=2 ,but your shapes is: "
                    << in_tensors_[0]->shape()[1] << "your input_dim_ is: " << input_dim_;
      return ERROR;
    }
  }
  auto param = reinterpret_cast<SparseToDenseParameter *>(op_parameter_);
  if (param->validate_indices_) {
    MS_LOG(ERROR) << "Unsupported unordered for in_tensors_indices";
    return RET_ERROR;
  }
  return RET_OK;
}

void SparseToDenseOpenCLKernel::SetConstArgs() {
  auto runtime_wrapper = lite::opencl::OpenCLRuntimeWrapper();
  GpuTensorInfo img_info(out_tensors_[0]);
  size_t dtype = enable_fp16_ ? sizeof(cl_half) : sizeof(cl_float);
  stride_w = img_info.RowPitch() / dtype;
  cl_int2 input_shape = {n_ * h_, w_ * UP_DIV(c_, C4NUM)};
  auto out_shape_temp = out_tensors_[0]->shape();
  cl_int4 out_shape = {out_n_, out_h_, out_w_, UP_DIV(out_c_, C4NUM)};
  int arg_cn = 3;
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, input_shape);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_shape);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, default_);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, stride_w);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, inshapeindex1_dim);
}

void SparseToDenseOpenCLKernel::SetGlobalLocal() {
  local_size_ = {1, 1};
  size_t OH = n_ * h_;
  size_t OW = w_ * UP_DIV(c_, C4NUM);
  global_size_ = {OH, OW};
  OpenCLKernel::AlignGlobalLocal(global_size_, local_size_);
}

int SparseToDenseOpenCLKernel::Prepare() {
  input_dim_ = in_tensors_[0]->shape().size();
  inshapeindex1_dim = in_tensors_[0]->shape()[1];
  weight_scalar_ = in_tensors_[2]->IsScalar();
  std::string kernel_name = "SparseToDense" + std::string(weight_scalar_ ? "Scalar" : "Vector");
  std::string source = sparse_to_dense_source;
  std::string program_name = "SparseToDense";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);

  if (in_tensors_.size() > 3) {
    auto input_tensor3 = in_tensors_[3];
    if (input_tensor3->data_type() == kNumberTypeFloat16) {
      default_ = static_cast<float>(*reinterpret_cast<float16_t *>(input_tensor3->data_c()));
    } else {
      default_ = *reinterpret_cast<float *>(input_tensor3->data_c());
    }
  }
  InitWeights();
  InferShapeTo4D();
  SetGlobalLocal();
  SetConstArgs();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int SparseToDenseOpenCLKernel::InferShapeTo4D() {
  if (in_tensors_[0]->shape().size() <= 4) {
    if (in_tensors_[0]->shape().size() == 1) {
      n_ = in_tensors_[0]->shape()[0];
    } else if (in_tensors_[0]->shape().size() == 2) {
      n_ = in_tensors_[0]->shape()[0];
      c_ = in_tensors_[0]->shape()[1];
    }
  }
  if (out_tensors_[0]->shape().size() <= 4) {
    if (out_tensors_[0]->shape().size() == 1) {
      out_n_ = out_tensors_[0]->shape()[0];
    } else if (out_tensors_[0]->shape().size() == 2) {
      out_n_ = out_tensors_[0]->shape()[0];
      out_c_ = out_tensors_[0]->shape()[1];
    } else if (out_tensors_[0]->shape().size() == 3) {
      out_n_ = out_tensors_[0]->shape()[0];
      out_w_ = out_tensors_[0]->shape()[1];
      out_c_ = out_tensors_[0]->shape()[2];
    } else {
      out_n_ = out_tensors_[0]->shape()[0];
      out_h_ = out_tensors_[0]->shape()[1];
      out_w_ = out_tensors_[0]->shape()[2];
      out_c_ = out_tensors_[0]->shape()[3];
    }
  }
  return RET_OK;
}

int SparseToDenseOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  InitOutputToDefault();
  int arg_cn = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_tensors_[0]->data_c(), lite::opencl::MemType::BUF);
  if (!weight_scalar_) {
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, weight_vector_, lite::opencl::MemType::BUF);
  } else {
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, weight_scalar_);
  }
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_SparseToDense, OpenCLKernelCreator<SparseToDenseOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_SparseToDense, OpenCLKernelCreator<SparseToDenseOpenCLKernel>);
}  // namespace mindspore::kernel
