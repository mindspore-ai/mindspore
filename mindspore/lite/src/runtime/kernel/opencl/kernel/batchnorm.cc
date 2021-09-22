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
#include "src/runtime/kernel/opencl/kernel/batchnorm.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/kernel/opencl/cl/batchnorm.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BatchNorm;
namespace {
constexpr int kNumInput0 = 0;
constexpr int kNumInput1 = 1;
constexpr int kNumInput2 = 2;
constexpr int kNumInput3 = 3;
constexpr int kNumInput4 = 4;
}  // namespace
namespace mindspore::kernel {
int BatchNormOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_5 || out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  if (in_tensors_.at(0)->shape().size() != DIMENSION_4D) {
    MS_LOG(WARNING) << "The dim of in_tensors->shape must be 4 but your dim is : " << in_tensors_.at(0)->shape().size();
    return RET_ERROR;
  }
  if (in_tensors_.at(0)->shape()[0] > 1) {
    MS_LOG(WARNING) << "  Unsupported batch_size >1 ";
    return RET_ERROR;
  }
  CHECK_NULL_RETURN(in_tensors_[kNumInput0]);
  CHECK_NULL_RETURN(in_tensors_[kNumInput1]);
  CHECK_NULL_RETURN(in_tensors_[kNumInput2]);
  CHECK_NULL_RETURN(in_tensors_[kNumInput3]);
  CHECK_NULL_RETURN(in_tensors_[kNumInput4]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  return RET_OK;
}

void BatchNormGetWorkGroup(const std::vector<size_t> &global, std::vector<size_t> *local, int max_size) {
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

int BatchNormOpenCLKernel::SetConstArgs() {
  int arg_cn = 6;
  auto param = reinterpret_cast<BatchNormParameter *>(this->op_parameter_);
  auto input0_shape = in_tensors_.at(0)->shape();
  cl_int4 input_shape_ = {input0_shape.at(0), input0_shape.at(1), input0_shape.at(2),
                          UP_DIV(input0_shape.at(3), C4NUM)};
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, input_shape_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, param->epsilon_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, input0_shape.at(3)) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

void BatchNormOpenCLKernel::SetGlobalLocal() {
  auto output_shape = out_tensors_.at(0)->shape();
  uint32_t OH = output_shape.at(1);
  uint32_t OW = output_shape.at(2);
  uint32_t OC = UP_DIV(output_shape.at(3), C4NUM);

  const std::vector<size_t> &max_global = ocl_runtime_->GetWorkItemSize();
  local_size_ = {1, 1, 1};  // init local
  global_size_ = {OH, OW, OC};
  BatchNormGetWorkGroup(global_size_, &local_size_, max_global[0]);
  OpenCLKernel::AlignGlobalLocal(global_size_, local_size_);
}

int BatchNormOpenCLKernel::UnmapBuffer() {
  auto allocator = ocl_runtime_->GetAllocator();
  if (allocator->UnmapBuffer(scale_) != RET_OK) {
    return RET_ERROR;
  }
  if (allocator->UnmapBuffer(offset_) != RET_OK) {
    return RET_ERROR;
  }
  if (allocator->UnmapBuffer(mean_) != RET_OK) {
    return RET_ERROR;
  }
  if (allocator->UnmapBuffer(variance_) != RET_OK) {
    return RET_ERROR;
  }
  return RET_OK;
}

int BatchNormOpenCLKernel::MapBuffer() {
  auto allocator = ocl_runtime_->GetAllocator();
  if (allocator->MapBuffer(scale_, CL_MAP_WRITE, nullptr, true) == nullptr) {
    return RET_ERROR;
  }
  if (allocator->MapBuffer(offset_, CL_MAP_WRITE, nullptr, true) == nullptr) {
    return RET_ERROR;
  }
  if (allocator->MapBuffer(mean_, CL_MAP_WRITE, nullptr, true) == nullptr) {
    return RET_ERROR;
  }
  if (allocator->MapBuffer(variance_, CL_MAP_WRITE, nullptr, true) == nullptr) {
    return RET_ERROR;
  }

  return RET_OK;
}

int BatchNormOpenCLKernel::Initweight() {
  auto allocator = ocl_runtime_->GetAllocator();
  GpuTensorInfo img_info(in_tensors_.at(1));
  auto weight_tensor = in_tensors_.at(1);
  size_t weight_size = img_info.OriginSize;
  // allocated memory for weight and init value
  scale_ = allocator->Malloc(weight_size, lite::opencl::MemType::BUF);
  if (scale_ == nullptr) {
    MS_LOG(ERROR) << "Malloc failed.";
    return RET_ERROR;
  }
  offset_ = allocator->Malloc(weight_size, lite::opencl::MemType::BUF);
  if (offset_ == nullptr) {
    MS_LOG(ERROR) << "Malloc failed.";
    return RET_ERROR;
  }
  mean_ = allocator->Malloc(weight_size, lite::opencl::MemType::BUF);
  if (mean_ == nullptr) {
    MS_LOG(ERROR) << "Malloc failed.";
    return RET_ERROR;
  }
  variance_ = allocator->Malloc(weight_size, lite::opencl::MemType::BUF);
  if (variance_ == nullptr) {
    MS_LOG(ERROR) << "Malloc failed.";
    return RET_ERROR;
  }

  if (MapBuffer() != RET_OK) {
    MS_LOG(ERROR) << "Map Buffer failed.";
    return RET_ERROR;
  }
  memset(scale_, 1, weight_size);
  memset(offset_, 0x00, weight_size);
  memset(mean_, 0x00, weight_size);
  memset(variance_, 0x00, weight_size);
  CHECK_NULL_RETURN(in_tensors_.at(kNumInput1)->data());
  CHECK_NULL_RETURN(in_tensors_.at(kNumInput2)->data());
  CHECK_NULL_RETURN(in_tensors_.at(kNumInput3)->data());
  CHECK_NULL_RETURN(in_tensors_.at(kNumInput4)->data());
  if (weight_tensor->data_type() == kNumberTypeFloat16) {
    if (use_fp16_enable_) {
      memcpy(scale_, in_tensors_.at(1)->data(), weight_size);
      memcpy(offset_, in_tensors_.at(2)->data(), weight_size);
      memcpy(mean_, in_tensors_.at(3)->data(), weight_size);
      memcpy(variance_, in_tensors_.at(4)->data(), weight_size);
    } else {
      auto scale_fp32 = reinterpret_cast<float *>(scale_);
      auto offset_fp32 = reinterpret_cast<float *>(offset_);
      auto mean_fp32 = reinterpret_cast<float *>(mean_);
      auto variance_fp32 = reinterpret_cast<float *>(variance_);

      auto origin_scale_fp16 = reinterpret_cast<float16_t *>(in_tensors_.at(1)->data());
      auto origin_offset_fp16 = reinterpret_cast<float16_t *>(in_tensors_.at(2)->data());
      auto origin_mean_fp16 = reinterpret_cast<float16_t *>(in_tensors_.at(3)->data());
      auto origin_variance_fp16 = reinterpret_cast<float16_t *>(in_tensors_.at(4)->data());

      for (int i = 0; i < img_info.ElementsNum; ++i) {
        scale_fp32[i] = static_cast<float>(origin_scale_fp16[i]);
        offset_fp32[i] = static_cast<float>(origin_offset_fp16[i]);
        mean_fp32[i] = static_cast<float>(origin_mean_fp16[i]);
        variance_fp32[i] = static_cast<float>(origin_variance_fp16[i]);
      }
    }
  } else {
    if (use_fp16_enable_) {
      auto scale_fp16 = reinterpret_cast<float16_t *>(scale_);
      auto offset_fp16 = reinterpret_cast<float16_t *>(offset_);
      auto mean_fp16 = reinterpret_cast<float16_t *>(mean_);
      auto variance_fp16 = reinterpret_cast<float16_t *>(variance_);

      auto origin_scale_fp32 = reinterpret_cast<float *>(in_tensors_.at(1)->data());
      auto origin_offset_fp32 = reinterpret_cast<float *>(in_tensors_.at(2)->data());
      auto origin_mean_fp32 = reinterpret_cast<float *>(in_tensors_.at(3)->data());
      auto origin_variance_fp32 = reinterpret_cast<float *>(in_tensors_.at(4)->data());

      for (int i = 0; i < img_info.ElementsNum; ++i) {
        scale_fp16[i] = static_cast<float16_t>(origin_scale_fp32[i]);
        offset_fp16[i] = static_cast<float16_t>(origin_offset_fp32[i]);
        mean_fp16[i] = static_cast<float16_t>(origin_mean_fp32[i]);
        variance_fp16[i] = static_cast<float16_t>(origin_variance_fp32[i]);
      }
    } else {
      memcpy(scale_, in_tensors_.at(1)->data(), weight_size);
      memcpy(offset_, in_tensors_.at(2)->data(), weight_size);
      memcpy(mean_, in_tensors_.at(3)->data(), weight_size);
      memcpy(variance_, in_tensors_.at(4)->data(), weight_size);
    }
  }
  if (UnmapBuffer() != RET_OK) {
    MS_LOG(ERROR) << "UnmapBuffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int BatchNormOpenCLKernel::Prepare() {
  use_fp16_enable_ = ocl_runtime_->GetFp16Enable();
  const std::string kernel_name = "Batch_normalization_NHWC4";
  std::string source = batchnorm_source;
  const std::string program_name = "Batch_normalization";
  if (!ocl_runtime_->LoadSource(program_name, source)) {
    MS_LOG(ERROR) << "Load source failed.";
    return RET_ERROR;
  }
  auto build_options_ext = CreateBuildOptionsExtByDType(this->registry_data_type_);
  auto ret = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options_ext);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build kernel failed.";
    return ret;
  }
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  ret = Initweight();
  if (ret) {
    MS_LOG(ERROR) << "Initweight failed ";
    return RET_ERROR;
  }
  if (SetConstArgs() != RET_OK) {
    MS_LOG(ERROR) << "SeConstArgs failed.";
    return RET_ERROR;
  }
  SetGlobalLocal();

  return RET_OK;
}

int BatchNormOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  int arg_cn = 0;
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_tensors_.at(0)->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }  // input tensor
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, scale_, true) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }  // scale
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, offset_, true) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }  // offset
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, mean_, true) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }  // mean
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, variance_, true) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }  // variance
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_tensors_.at(0)->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }  // out tensor
  if (ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_BatchNorm, OpenCLKernelCreator<BatchNormOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_BatchNorm, OpenCLKernelCreator<BatchNormOpenCLKernel>)
}  // namespace mindspore::kernel
