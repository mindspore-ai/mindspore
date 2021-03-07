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

#include "src/runtime/kernel/opencl/kernel/prelu.h"
#include <mindspore/lite/nnacl/prelu_parameter.h>
#include <set>
#include <vector>
#include "src/runtime/kernel/opencl/cl/prelu.cl.inc"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp32/common_func_fp32.h"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_PReLUFusion;

namespace mindspore::kernel {

int PReluOpenCLKernel::InitWeights() {
  auto allocator = ocl_runtime_->GetAllocator();
  auto weight_tensor = in_tensors_.at(1);
  if (weight_is_scalar) {
    if (weight_tensor->data_type() == kNumberTypeFloat16) {
      weight_scalar_ = static_cast<float>(*reinterpret_cast<float16_t *>(weight_tensor->data_c()));
    } else {
      weight_scalar_ = *reinterpret_cast<float *>(weight_tensor->data_c());
    }
  } else {
    int C_ = weight_tensor->ElementsNum();
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

int PReluOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() != 2 || out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "PRelu Only supported in_tensors_.size=2 and out_tensors_.size()=1 but your in_tensors_.size="
                  << in_tensors_.size() << " out_tensors_.size()=" << out_tensors_.size();
    return RET_ERROR;
  }
  auto weight_tensor = in_tensors_.at(1);
  auto in_tensor_channel = GpuTensorInfo(in_tensors_[0]).C;
  auto weight_channel = GpuTensorInfo(in_tensors_[1]).C;
  if (weight_channel != 1 && weight_channel != in_tensor_channel) {
    MS_LOG(ERROR) << "PRelu weight must be equal with in_teneors channel size, but your weight size is "
                  << weight_channel << " and your input channel size is " << in_tensor_channel;
    return mindspore::lite::RET_ERROR;
  }
  if (weight_tensor->data_type() != kNumberTypeFloat16 && weight_tensor->data_type() != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "PRelu weight must be float32 or float16";
    return RET_ERROR;
  }
  return RET_OK;
}

void PReluOpenCLKernel::SetConstArgs() {
  int arg_idx = 3;
  out_shape_.s[3] = UP_DIV(out_shape_.s[3], C4NUM);
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_shape_);
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, 2);
}

void PReluOpenCLKernel::SetGlobalLocal() {
  std::vector<size_t> local = {4, 4, 1};
  OH = out_shape_.s[0] * out_shape_.s[1];
  OW = out_shape_.s[2];
  OC = out_shape_.s[3];
  std::vector<size_t> global = {OH, OW, OC};
  AlignGlobalLocal(global, local);
}

int PReluOpenCLKernel::Prepare() {
  cl_int4 output_shape = {};
  cl_int4 weight_shape = {};
  for (int i = 0; i < out_tensors_.at(0)->shape().size(); ++i) {
    output_shape.s[i] = out_tensors_.at(0)->shape()[i];
  }
  for (int i = 0; i < in_tensors_.at(1)->shape().size(); ++i) {
    weight_shape.s[i] = in_tensors_.at(1)->shape()[i];
  }
  Broadcast2GpuShape(out_shape_.s, output_shape.s, out_tensors_.at(0)->shape().size(), 1);
  Broadcast2GpuShape(weight_shape_.s, weight_shape.s, in_tensors_.at(1)->shape().size(), 1);
  auto param = reinterpret_cast<PReluParameter *>(op_parameter_);
  weight_is_scalar = param->channelShared;
  enable_fp16_ = ocl_runtime_->GetFp16Enable();
  std::string source = prelu_source;
  std::string program_name = "PRelu";
  std::string kernel_name = "PRelu_" + std::string(weight_is_scalar ? "scalar" : "vector");
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);
  InitWeights();
  MS_LOG(DEBUG) << program_name << " init Done!";
  MS_LOG(DEBUG) << "kernel_name=: " << kernel_name << " init Done!";
  SetConstArgs();
  SetGlobalLocal();
  return mindspore::lite::RET_OK;
}

int PReluOpenCLKernel::Run() {
  MS_LOG(DEBUG) << op_parameter_->name_ << " Running!";
  int arg_idx = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->data_c());
  if (weight_is_scalar) {
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, weight_scalar_);
  } else {
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, weight_vector_, lite::opencl::MemType::BUF);
  }
  auto ret = ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "Run kernel " << op_parameter_->name_ << " error.";
    return mindspore::lite::RET_ERROR;
  }
  return mindspore::lite::RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_PReLUFusion, OpenCLKernelCreator<PReluOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_PReLUFusion, OpenCLKernelCreator<PReluOpenCLKernel>)
}  // namespace mindspore::kernel
