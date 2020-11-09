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
using mindspore::schema::PrimitiveType_SparseToDense;

namespace mindspore::kernel {

int SparseToDenseOpenCLKernel::InitOutputToDefault() {
  auto allocator_ = ocl_runtime_->GetAllocator();
  std::vector<size_t> img_size;
  cl_float4 fill_value = {};
  fill_value.s[0] = fill_value.s[1] = fill_value.s[2] = fill_value.s[3] = default_;
  auto src_data = out_tensors_[0]->data_c();
  allocator_->GetImageSize(src_data, &img_size);
  auto src_origin = cl::array<cl::size_type, 3U>{0, 0, 0};
  auto region = cl::array<cl::size_type, 3U>{img_size[0], img_size[1], 1};
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

int SparseToDenseOpenCLKernel::Init() {
  if (out_tensors_[0]->shape().size() > 2 || in_tensors_.size() < 3) {
    MS_LOG(ERROR) << " only support dim <= 2 and in_tensors_.size >= 3";
    return RET_ERROR;
  }
  if ((in_tensors_[0]->shape()[1] > 3) && (input_dim_ == 2)) {
    MS_LOG(ERROR) << "in_tensors_indices shape[1] must be 1 2 or 3  && input_dim_=2 ,but your shapes is: "
                  << in_tensors_[0]->shape()[1] << "your input_dim_ is: " << input_dim_;
    return ERROR;
  }
  input_dim_ = in_tensors_[0]->shape().size();
  weight_scalar_ = in_tensors_[2]->IsScalar();
  std::string kernel_name = "SparseToDense" + std::string(weight_scalar_ ? "ScalarDim" : "VectorDim") +
                            std::to_string(in_tensors_[0]->shape()[1] == 1 ? 1 : input_dim_);
  if (input_dim_ == 2 && in_tensors_[0]->shape()[1] != 1) {
    kernel_name += "Shape" + std::to_string(in_tensors_[0]->shape()[1]);
  }

  std::set<std::string> build_options;
  std::string source = sparse_to_dense_source;
  std::string program_name = "SparseToDense";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);

  if (in_tensors_.size() > 3) {
    auto input_tensor3 = in_tensors_[3];
    if (input_tensor3->data_type() == kNumberTypeFloat16) {
      default_ = static_cast<float>(*reinterpret_cast<float16_t *>(input_tensor3->data_c()));
    } else {
      default_ = *reinterpret_cast<float *>(input_tensor3->data_c());
    }
  }

  InitWeights();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int SparseToDenseOpenCLKernel::InferShapeTo4D() {
  if (in_tensors_[0]->shape().size() <= 4) {
    if (in_tensors_[0]->shape().size() == 1) {
      N_ = in_tensors_[0]->shape()[0];
    } else if (in_tensors_[0]->shape().size() == 2) {
      N_ = in_tensors_[0]->shape()[0];
      C_ = in_tensors_[0]->shape()[1];
    } else if (in_tensors_[0]->shape().size() == 3) {
      N_ = in_tensors_[0]->shape()[0];
      W_ = in_tensors_[0]->shape()[1];
      C_ = in_tensors_[0]->shape()[2];
    } else {
      N_ = in_tensors_[0]->shape()[0];
      H_ = in_tensors_[0]->shape()[1];
      W_ = in_tensors_[0]->shape()[2];
      C_ = in_tensors_[0]->shape()[3];
    }
  } else {
    MS_LOG(ERROR) << "Unsupported inputdim: " << in_tensors_[0]->shape().size();
    return RET_ERROR;
  }
  return RET_OK;
}

int SparseToDenseOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  InferShapeTo4D();
  cl_int2 input_shape = {static_cast<cl_int>(N_ * H_), static_cast<cl_int>(W_ * UP_DIV(C_, C4NUM))};
  InitOutputToDefault();
  std::vector<size_t> local = {1, 1};
  std::vector<size_t> global = {1, 1};
  int arg_cn = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_tensors_[0]->data_c());
  if (weight_scalar_) {
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, weight_scalar_);
  } else {
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, weight_vector_);
  }
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, input_shape);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, default_);
  ocl_runtime_->RunKernel(kernel_, global, local, nullptr);
  return RET_OK;
}

kernel::LiteKernel *SparseToDenseOpenCLKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                     const std::vector<lite::Tensor *> &outputs,
                                                     OpParameter *opParameter, const lite::InnerContext *ctx,
                                                     const kernel::KernelKey &desc,
                                                     const mindspore::lite::PrimitiveC *primitive) {
  if (inputs.empty()) {
    MS_LOG(ERROR) << "Input data size must be greater than 0, but your size is " << inputs.size();
    free(opParameter);
    return nullptr;
  }
  auto *kernel = new (std::nothrow) SparseToDenseOpenCLKernel(opParameter, inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << " new HswishOpenCLKernel failed ";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << " Init kernel failed, name: hswish ";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_SparseToDense, SparseToDenseOpenCLKernelCreator);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_SparseToDense, SparseToDenseOpenCLKernelCreator);
}  // namespace mindspore::kernel
