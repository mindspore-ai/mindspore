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
#include <cstring>
#include <algorithm>
#include <set>
#include <string>
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/kernel/layer_norm.h"
#include "nnacl/layer_norm_parameter.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/kernel/opencl/cl/layer_norm.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LayerNormFusion;

namespace mindspore::kernel {

int LayerNormOpenCLKernel::CheckSpecs() {
  auto param = reinterpret_cast<LayerNormParameter *>(this->op_parameter_);
  if (in_tensors_.size() != 3 || out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "UnSupported in_tensors_.size: " << in_tensors_.size()
                  << " out_tensors_.size(): " << out_tensors_.size();
    return RET_ERROR;
  }
  if (in_tensors_.at(0)->shape().size() != 4) {
    MS_LOG(ERROR) << "UnSupported in_tensors_.shape.size: " << in_tensors_.at(0)->shape().size();
    return RET_ERROR;
  }
  normalized_axis_ = param->begin_params_axis_;
  epsilon_ = param->epsilon_;
  if (normalized_axis_ < 0) {
    normalized_axis_ += in_tensors_.at(0)->shape().size();
  }
  if (normalized_axis_ != 3) {
    MS_LOG(ERROR) << "UnSupported normalized_axis_ : " << param->normalized_dims_;
    return RET_ERROR;
  }
  return RET_OK;
}

void LayerNormGetWorkGroup(const std::vector<size_t> &global, std::vector<size_t> *local, int max_size) {
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

void LayerNormOpenCLKernel::SetConstArgs() {
  int arg_cn = 6;
  GpuTensorInfo img_info(in_tensors_.at(0));
  in_shape_.s[0] = img_info.N, in_shape_.s[1] = img_info.H, in_shape_.s[2] = img_info.W, in_shape_.s[3] = img_info.C;
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_shape_);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, epsilon_);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, normalized_axis_);
  ocl_runtime_->SetKernelArg(kernel_mean_var_, 3, in_shape_);
  ocl_runtime_->SetKernelArg(kernel_mean_var_, 4, normalized_shape_size_);
}

void AlignMeanVarGlobalLocal(const std::vector<int> &global, const std::vector<int> &local, cl::NDRange *global_range,
                             cl::NDRange *local_range) {
  *local_range = cl::NDRange(local[0], local[1], local[2]);
  *global_range =
    cl::NDRange(UP_ROUND(global[0], local[0]), UP_ROUND(global[1], local[1]), UP_ROUND(global[2], local[2]));
}

void LayerNormOpenCLKernel::SetGlobalLocal() {
  size_t OH = 1, OW = 1, OC = 1;
  OH = in_shape_.s[0] * in_shape_.s[1];
  OW = in_shape_.s[2];
  OC = UP_DIV(in_shape_.s[3], C4NUM);
  local_size_ = {1, 1, 1};  // init local
  global_size_ = {OH, OW, OC};
  const std::vector<size_t> &max_global = ocl_runtime_->GetWorkItemSize();
  LayerNormGetWorkGroup(global_size_, &local_size_, max_global[0]);
  OpenCLKernel::AlignGlobalLocal(global_size_, local_size_);
  AlignMeanVarGlobalLocal({static_cast<int>(OH), static_cast<int>(OW), 1}, {1, 1, 1}, &global_mean_var_,
                          &local_mean_var_);
}

int LayerNormOpenCLKernel::Initweight() {
  auto allocator = ocl_runtime_->GetAllocator();
  GpuTensorInfo img_info(in_tensors_.at(1));
  auto weight_tensor = in_tensors_.at(1);
  size_t weight_size = img_info.Image2DSize;
  // allocated memory for weight and init value
  gamma_ = allocator->Malloc(weight_size);
  beta_ = allocator->Malloc(weight_size);
  allocator->MapBuffer(gamma_, CL_MAP_WRITE, nullptr, true);
  allocator->MapBuffer(beta_, CL_MAP_WRITE, nullptr, true);
  memset(gamma_, 0x01, weight_size);
  memset(beta_, 0x00, weight_size);

  if (weight_tensor->data_type() == kNumberTypeFloat16) {
    if (use_fp16_enable_) {
      memcpy(gamma_, in_tensors_.at(1)->data_c(), weight_size);
      memcpy(beta_, in_tensors_.at(2)->data_c(), weight_size);
    } else {
      auto gamma_fp32 = reinterpret_cast<float *>(gamma_);
      auto beta_fp32 = reinterpret_cast<float *>(beta_);
      auto origin_gamma_fp16 = reinterpret_cast<float16_t *>(in_tensors_.at(1)->data_c());
      auto origin_beta_fp16 = reinterpret_cast<float16_t *>(in_tensors_.at(2)->data_c());

      for (int i = 0; i < img_info.ElementsNum; ++i) {
        gamma_fp32[i] = static_cast<float>(origin_gamma_fp16[i]);
        beta_fp32[i] = static_cast<float>(origin_beta_fp16[i]);
      }
    }
  } else {
    if (use_fp16_enable_) {
      auto gamma_fp16 = reinterpret_cast<float16_t *>(gamma_);
      auto beta_fp16 = reinterpret_cast<float16_t *>(beta_);
      auto origin_gamma_fp32 = reinterpret_cast<float *>(in_tensors_.at(1)->data_c());
      auto origin_beta_fp32 = reinterpret_cast<float *>(in_tensors_.at(2)->data_c());

      for (int i = 0; i < img_info.ElementsNum; ++i) {
        gamma_fp16[i] = static_cast<float16_t>(origin_gamma_fp32[i]);
        beta_fp16[i] = static_cast<float16_t>(origin_beta_fp32[i]);
      }
    } else {
      memcpy(gamma_, in_tensors_.at(1)->data_c(), weight_size);
      memcpy(beta_, in_tensors_.at(2)->data_c(), weight_size);
    }
  }
  allocator->UnmapBuffer(gamma_);
  allocator->UnmapBuffer(beta_);
  return RET_OK;
}

int LayerNormOpenCLKernel::Prepare() {
  use_fp16_enable_ = ocl_runtime_->GetFp16Enable();
  int ret = Initweight();
  if (ret) {
    MS_LOG(ERROR) << "Initweight failed ";
    return RET_ERROR;
  }
  normalized_shape_size_ = in_tensors_.at(0)->shape().at(normalized_axis_);
  auto allocator = ocl_runtime_->GetAllocator();
  size_t mean_size = 1;
  for (int i = 0; i < normalized_axis_; ++i) {
    mean_size *= in_tensors_.at(0)->shape()[i];
  }
  size_t size_dtype = use_fp16_enable_ ? sizeof(float16_t) : sizeof(float);
  mean_size *= size_dtype;
  mean_ = allocator->Malloc(mean_size);
  var_ = allocator->Malloc(mean_size);
  std::string kernel_name = "LayerNormalization_NHWC4";
  std::string kernel_name_mean_var = "ComputeMeanVar";
  std::string source = layer_norm_source;
  std::string program_name = "LayerNormalization";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);
  kernel_name_mean_var += "Axis" + std::to_string(normalized_axis_) + "NHWC4";
  ocl_runtime_->BuildKernel(kernel_mean_var_, program_name, kernel_name_mean_var);
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  SetConstArgs();
  SetGlobalLocal();

  return RET_OK;
}

int LayerNormOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  int arg1_cn = 0;
  ocl_runtime_->SetKernelArg(kernel_mean_var_, arg1_cn++, in_tensors_.at(0)->data_c());        // input tensor
  ocl_runtime_->SetKernelArg(kernel_mean_var_, arg1_cn++, mean_, lite::opencl::MemType::BUF);  // mean_
  ocl_runtime_->SetKernelArg(kernel_mean_var_, arg1_cn++, var_, lite::opencl::MemType::BUF);   // var_  return RET_OK;
  ocl_runtime_->RunKernel(kernel_mean_var_, global_mean_var_, local_mean_var_, nullptr, &event_);

  int arg_cn = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_tensors_.at(0)->data_c());         // input tensor
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_tensors_.at(0)->data_c());        // out tensor
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, mean_, lite::opencl::MemType::BUF);   // mean_
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, var_, lite::opencl::MemType::BUF);    // var_
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, gamma_, lite::opencl::MemType::BUF);  // gamma_
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, beta_, lite::opencl::MemType::BUF);   // beta_
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_LayerNormFusion, OpenCLKernelCreator<LayerNormOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_LayerNormFusion, OpenCLKernelCreator<LayerNormOpenCLKernel>)
}  // namespace mindspore::kernel
