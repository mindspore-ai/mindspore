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
  CHECK_NULL_RETURN(param);
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_3 || out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "UnSupported in_tensors_.size: " << in_tensors_.size()
                    << " out_tensors_.size(): " << out_tensors_.size();
    return RET_ERROR;
  }
  auto *input = in_tensors_.at(0);
  CHECK_NULL_RETURN(input);
  auto *output = out_tensors_.at(0);
  CHECK_NULL_RETURN(output);
  if (input->shape().size() != DIMENSION_4D) {
    MS_LOG(WARNING) << "UnSupported in_tensors_.shape.size: " << input->shape().size();
    return RET_ERROR;
  }
  normalized_axis_ = param->begin_params_axis_;
  epsilon_ = param->epsilon_;
  if (normalized_axis_ < 0) {
    normalized_axis_ += input->shape().size();
  }
  if (normalized_axis_ != 3) {
    MS_LOG(WARNING) << "UnSupported normalized_axis_ : " << param->normalized_dims_;
    return RET_ERROR;
  }
  return RET_OK;
}

void LayerNormGetWorkGroup(const std::vector<size_t> &global, std::vector<size_t> *local, int max_size) {
  const int max_divider = 8;
  const int max_x = 4, max_y = 8;
  int x = std::min(GetMaxDivisorStrategy1(global[0], max_divider), max_x);
  MS_ASSERT(x);
  int yz = max_size / x;
  int y = std::min(std::min(GetMaxDivisorStrategy1(global[1], max_divider), yz), max_y);
  MS_ASSERT(y);
  int z = std::min(yz / y, static_cast<int>(UP_DIV(global[2], 2)));

  local->clear();
  local->push_back(x);
  local->push_back(y);
  local->push_back(z);
}

int LayerNormOpenCLKernel::SetConstArgs() {
  int arg_cn = 6;
  GpuTensorInfo img_info(in_tensors_.at(0));
  in_shape_.s[0] = img_info.N, in_shape_.s[1] = img_info.H, in_shape_.s[2] = img_info.W, in_shape_.s[3] = img_info.C;
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_shape_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, epsilon_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, normalized_axis_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_mean_var_, 3, in_shape_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_mean_var_, 4, normalized_shape_size_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  return RET_OK;
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
  CHECK_NULL_RETURN(allocator);
  GpuTensorInfo img_info(in_tensors_.at(1));
  auto weight_tensor = in_tensors_.at(1);
  CHECK_NULL_RETURN(weight_tensor);
  size_t weight_size = img_info.Image2DSize;
  // allocated memory for weight and init value
  gamma_ = allocator->Malloc(weight_size, lite::opencl::MemType::BUF);
  if (gamma_ == nullptr) {
    MS_LOG(ERROR) << "Malloc failed.";
    return RET_ERROR;
  }
  beta_ = allocator->Malloc(weight_size, lite::opencl::MemType::BUF);
  if (beta_ == nullptr) {
    MS_LOG(ERROR) << "Malloc failed.";
    return RET_ERROR;
  }
  if (allocator->MapBuffer(gamma_, CL_MAP_WRITE, nullptr, true) == nullptr) {
    MS_LOG(ERROR) << "Map Buffer failed.";
    return RET_ERROR;
  }
  if (allocator->MapBuffer(beta_, CL_MAP_WRITE, nullptr, true) == nullptr) {
    MS_LOG(ERROR) << "Map Buffer failed.";
    return RET_ERROR;
  }
  memset(gamma_, 0x01, weight_size);
  memset(beta_, 0x00, weight_size);
  CHECK_NULL_RETURN(in_tensors_.at(1)->data());
  CHECK_NULL_RETURN(in_tensors_.at(2));
  CHECK_NULL_RETURN(in_tensors_.at(2)->data());

  if (weight_tensor->data_type() == kNumberTypeFloat16) {
    if (use_fp16_enable_) {
      memcpy(gamma_, in_tensors_.at(1)->data(), weight_size);
      memcpy(beta_, in_tensors_.at(2)->data(), weight_size);
    } else {
      auto gamma_fp32 = reinterpret_cast<float *>(gamma_);
      auto beta_fp32 = reinterpret_cast<float *>(beta_);
      auto origin_gamma_fp16 = reinterpret_cast<float16_t *>(in_tensors_.at(1)->data());
      auto origin_beta_fp16 = reinterpret_cast<float16_t *>(in_tensors_.at(2)->data());

      for (int i = 0; i < img_info.ElementsNum; ++i) {
        gamma_fp32[i] = static_cast<float>(origin_gamma_fp16[i]);
        beta_fp32[i] = static_cast<float>(origin_beta_fp16[i]);
      }
    }
  } else {
    if (use_fp16_enable_) {
      auto gamma_fp16 = reinterpret_cast<float16_t *>(gamma_);
      auto beta_fp16 = reinterpret_cast<float16_t *>(beta_);
      auto origin_gamma_fp32 = reinterpret_cast<float *>(in_tensors_.at(1)->data());
      auto origin_beta_fp32 = reinterpret_cast<float *>(in_tensors_.at(2)->data());

      for (int i = 0; i < img_info.ElementsNum; ++i) {
        gamma_fp16[i] = static_cast<float16_t>(origin_gamma_fp32[i]);
        beta_fp16[i] = static_cast<float16_t>(origin_beta_fp32[i]);
      }
    } else {
      memcpy(gamma_, in_tensors_.at(1)->data(), weight_size);
      memcpy(beta_, in_tensors_.at(2)->data(), weight_size);
    }
  }
  if (allocator->UnmapBuffer(gamma_) != RET_OK) {
    MS_LOG(ERROR) << "UnmapBuffer failed.";
    return RET_ERROR;
  }
  if (allocator->UnmapBuffer(beta_) != RET_OK) {
    MS_LOG(ERROR) << "UnmapBuffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int LayerNormOpenCLKernel::Prepare() {
  use_fp16_enable_ = ocl_runtime_->GetFp16Enable();
  int ret = Initweight();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Initweight failed ";
    return ret;
  }
  normalized_shape_size_ = in_tensors_.at(0)->shape().at(normalized_axis_);
  auto allocator = ocl_runtime_->GetAllocator();
  size_t mean_size = 1;
  for (int i = 0; i < normalized_axis_; ++i) {
    mean_size *= in_tensors_.at(0)->shape()[i];
  }
  size_t size_dtype = use_fp16_enable_ ? sizeof(float16_t) : sizeof(float);
  mean_size *= size_dtype;
  mean_ = allocator->Malloc(mean_size, lite::opencl::MemType::BUF);
  if (mean_ == nullptr) {
    MS_LOG(ERROR) << "Malloc failed.";
    return RET_ERROR;
  }
  var_ = allocator->Malloc(mean_size, lite::opencl::MemType::BUF);
  if (var_ == nullptr) {
    MS_LOG(ERROR) << "Malloc failed.";
    return RET_ERROR;
  }
  const std::string kernel_name = "LayerNormalization_NHWC4";
  std::string kernel_name_mean_var = "ComputeMeanVar";
  std::string source = layer_norm_source;
  const std::string program_name = "LayerNormalization";
  if (!ocl_runtime_->LoadSource(program_name, source)) {
    MS_LOG(ERROR) << "Load source failed.";
    return RET_ERROR;
  }
  auto build_options_ext = CreateBuildOptionsExtByDType(this->registry_data_type_);
  ret = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options_ext);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build kernel failed.";
    return ret;
  }
  kernel_name_mean_var += "Axis" + std::to_string(normalized_axis_) + "NHWC4";
  ocl_runtime_->BuildKernel(kernel_mean_var_, program_name, kernel_name_mean_var, build_options_ext);
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  if (SetConstArgs() != RET_OK) {
    MS_LOG(ERROR) << "SeConstArgs failed.";
    return RET_ERROR;
  }
  SetGlobalLocal();

  return RET_OK;
}

int LayerNormOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  int arg1_cn = 0;
  if (ocl_runtime_->SetKernelArg(kernel_mean_var_, arg1_cn++, in_tensors_.at(0)->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }  // input tensor
  if (ocl_runtime_->SetKernelArg(kernel_mean_var_, arg1_cn++, mean_, true) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_mean_var_, arg1_cn++, var_, true) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  ocl_runtime_->RunKernel(kernel_mean_var_, global_mean_var_, local_mean_var_, nullptr, &event_);

  int arg_cn = 0;
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_tensors_.at(0)->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }  // input tensor
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_tensors_.at(0)->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }  // out tensor
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, mean_, true) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }  // mean_
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, var_, true) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }  // var_
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, gamma_, true) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }  // gamma_
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, beta_, true) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }  // beta_
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return RET_OK;
}  // namespace mindspore::kernel

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_LayerNormFusion, OpenCLKernelCreator<LayerNormOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_LayerNormFusion, OpenCLKernelCreator<LayerNormOpenCLKernel>)
}  // namespace mindspore::kernel
