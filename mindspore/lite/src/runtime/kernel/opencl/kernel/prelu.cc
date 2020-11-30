/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <set>
#include <vector>

#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp32/common_func_fp32.h"
#include "src/runtime/kernel/opencl/kernel/prelu.h"
#include "src/runtime/kernel/opencl/cl/prelu.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_PReLU;

namespace mindspore::kernel {

int PReluOpenCLKernel::InitWeights() {
  auto allocator = ocl_runtime_->GetAllocator();
  auto weight_tensor = in_tensors_[1];
  if (weight_is_scalar) {
    if (weight_tensor->data_type() == kNumberTypeFloat16) {
      weight_scalar_ = static_cast<float>(*reinterpret_cast<float16_t *>(weight_tensor->data_c()));
    } else {
      weight_scalar_ = *reinterpret_cast<float *>(weight_tensor->data_c());
    }
  } else {
    auto sizeof_FLT = enable_fp16_ ? sizeof(float16_t) : sizeof(float);
    size_t weight_size = UP_ROUND(C_, C4NUM) * sizeof_FLT;
    weight_vector_ = allocator->Malloc(weight_size);
    allocator->MapBuffer(weight_vector_, CL_MAP_WRITE, nullptr, true);
    memset(weight_vector_, 0x00, weight_size);
    if (weight_tensor->data_type() == kNumberTypeFloat16) {
      if (enable_fp16_) {
        memcpy(weight_vector_, weight_tensor->data_c(), C_ * sizeof_FLT);
      } else {
        auto weight_fp32 = reinterpret_cast<float *>(weight_vector_);
        auto origin_bias_fp16 = reinterpret_cast<float16_t *>(weight_tensor->data_c());
        for (int i = 0; i < C_; ++i) {
          weight_fp32[i] = static_cast<float>(origin_bias_fp16[i]);
        }
      }
    } else {
      if (enable_fp16_) {
        auto weight_fp16 = reinterpret_cast<float16_t *>(weight_vector_);
        auto origin_bias_fp32 = reinterpret_cast<float *>(weight_tensor->data_c());
        for (int i = 0; i < C_; ++i) {
          weight_fp16[i] = static_cast<float16_t>(origin_bias_fp32[i]);
        }
      } else {
        memcpy(weight_vector_, weight_tensor->data_c(), C_ * sizeof_FLT);
      }
    }
    allocator->UnmapBuffer(weight_vector_);
  }
  return RET_OK;
}

int PReluOpenCLKernel::Init() {
  auto input_tensor = in_tensors_[0];
  auto weight_tensor = in_tensors_[1];
  if (input_tensor->shape().size() != 4) {
    MS_LOG(ERROR) << "PRelu only support dim=4, but your dim=" << input_tensor->shape().size();
    return mindspore::lite::RET_ERROR;
  }
  batch_size_ = input_tensor->Batch();
  C_ = input_tensor->Channel();
  H_ = input_tensor->Height();
  W_ = input_tensor->Width();
  if (batch_size_ != 1) {
    MS_LOG(ERROR) << "Init PRelu kernel failed: Unsupported multi-batch.";
    return RET_ERROR;
  }
  auto weight_channel = weight_tensor->shape()[0];
  if (weight_channel != 1 && weight_channel != C_) {
    MS_LOG(ERROR)
      << "PRelu weight channel size must be 1 or must be equal with in_teneors channel size, but your weight size is "
      << weight_channel << " and your input channel size is " << C_;
    return mindspore::lite::RET_ERROR;
  }
  weight_is_scalar = weight_channel == 1;
  if (weight_tensor->data_type() != kNumberTypeFloat16 && weight_tensor->data_type() != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "PRelu weight must be float32 or float16";
    return RET_ERROR;
  }
  enable_fp16_ = ocl_runtime_->GetFp16Enable();

  std::string source = prelu_source;
  std::string program_name = "PRelu";
  std::string kernel_name = "PRelu_" + std::string(weight_is_scalar ? "scalar" : "vector");
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);

  InitWeights();
  MS_LOG(DEBUG) << program_name << " init Done!";
  return mindspore::lite::RET_OK;
}

int PReluOpenCLKernel::Run() {
  MS_LOG(DEBUG) << op_parameter_->name_ << " Running!";
  auto CO_SLICES_ = UP_DIV(C_, C4NUM);
  cl_int4 shape = {batch_size_, H_, W_, CO_SLICES_};

  int arg_idx = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->data_c());
  if (weight_is_scalar) {
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, weight_scalar_);
  } else {
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, weight_vector_, lite::opencl::MemType::BUF);
  }
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, shape);
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, 2);

  std::vector<size_t> local = {4, 4, 1};
  std::vector<size_t> global = {static_cast<size_t>(H_), static_cast<size_t>(W_), static_cast<size_t>(CO_SLICES_)};
  AlignGlobalLocal(global, local);
  auto ret = ocl_runtime_->RunKernel(kernel_, global_range_, local_range_);
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "Run kernel " << op_parameter_->name_ << " error.";
    return mindspore::lite::RET_ERROR;
  }
  return mindspore::lite::RET_OK;
}

kernel::LiteKernel *OpenCLPReluKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                             const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                             const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                             const lite::PrimitiveC *primitive) {
  if (inputs.empty()) {
    MS_LOG(ERROR) << "Input data size must be greater than 0, but your size is " << inputs.size();
    free(opParameter);
    return nullptr;
  }
  auto *kernel = new (std::nothrow) PReluOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel " << opParameter->name_ << "is nullptr.";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "Init PRelu kernel failed!";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_PReLU, OpenCLPReluKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_PReLU, OpenCLPReluKernelCreator)
}  // namespace mindspore::kernel
