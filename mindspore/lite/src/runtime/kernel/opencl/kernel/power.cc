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

#include "src/runtime/kernel/opencl/kernel/power.h"
#include <cstring>
#include <string>
#include <algorithm>
#include <set>
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/kernel/opencl/cl/power.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_PowFusion;

namespace mindspore::kernel {
int PowerOpenCLKernel::CheckSpecs() {
  if ((in_tensors_.size() != INPUT_TENSOR_SIZE_1 && in_tensors_.size() != INPUT_TENSOR_SIZE_2) ||
      out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "in size: " << in_tensors_.size() << "out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  if (in_tensors_.size() == INPUT_TENSOR_SIZE_2 &&
      in_tensors_.at(0)->shape().size() != in_tensors_.at(1)->shape().size()) {
    MS_LOG(WARNING) << "Unsupported input->shape.size " << in_tensors_.at(0)->shape().size()
                    << "!=" << in_tensors_.at(1)->shape().size();
    return RET_ERROR;
  }
  if (in_tensors_.at(0)->shape().size() > DIMENSION_4D) {
    MS_LOG(WARNING) << "in_tensors_->shape.size must be less than 4";
    return RET_ERROR;
  }
  return RET_OK;
}

void PowerGetWorkGroup(const std::vector<size_t> &global, std::vector<size_t> *local, int max_size) {
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

int PowerOpenCLKernel::SetConstArgs() {
  float unalign_w = static_cast<float>(out_shape_.s[3]);
  out_shape_.s[3] = UP_DIV(out_shape_.s[3], C4NUM);
  int arg_cn = 2;
  if (!broadcast_) {
    arg_cn++;
    if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_shape_) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  } else {
    if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_shape_) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  }
  if (use_fp16_enable_) {
    auto x = static_cast<float16_t>(power_);
    auto y = static_cast<float16_t>(shift_);
    auto z = static_cast<float16_t>(scale_);
    auto w = static_cast<float16_t>(unalign_w);
    cl_half4 parameter = {*(reinterpret_cast<uint16_t *>(&x)), *(reinterpret_cast<uint16_t *>(&y)),
                          *(reinterpret_cast<uint16_t *>(&z)), *(reinterpret_cast<uint16_t *>(&w))};
    if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, parameter) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  } else {
    cl_float4 parameter = {power_, shift_, scale_, unalign_w};
    if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, parameter) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

void PowerOpenCLKernel::SetGlobalLocal() {
  cl_int4 output_shape = {};
  for (int i = 0; i < out_tensors_.at(0)->shape().size(); ++i) {
    output_shape.s[i] = out_tensors_.at(0)->shape()[i];
  }
  Broadcast2GpuShape(out_shape_.s, output_shape.s, out_tensors_.at(0)->shape().size(), 1);
  const std::vector<size_t> &max_global = ocl_runtime_->GetWorkItemSize();
  std::vector<size_t> local_size_ = {1, 1, 1};
  uint32_t OH = out_shape_.s[0] * out_shape_.s[1];
  uint32_t OW = out_shape_.s[2];
  uint32_t OC = UP_DIV(out_shape_.s[3], C4NUM);
  std::vector<size_t> global_size_ = {OH, OW, OC};
  PowerGetWorkGroup(global_size_, &local_size_, max_global[0]);
  OpenCLKernel::AlignGlobalLocal(global_size_, local_size_);
}

int PowerOpenCLKernel::Prepare() {
  if (in_tensors_.size() == INPUT_TENSOR_SIZE_1) {
    broadcast_ = true;
  }
  use_fp16_enable_ = ocl_runtime_->GetFp16Enable();
  auto param = reinterpret_cast<PowerParameter *>(this->op_parameter_);
  std::string kernel_name = "power";
  std::string source = power_source;
  const std::string program_name = "power";
  if (broadcast_) {
    power_ = param->power_;
    kernel_name += "_broadcast";
  }
  scale_ = param->scale_;
  shift_ = param->shift_;
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
  SetGlobalLocal();
  if (SetConstArgs() != RET_OK) {
    MS_LOG(ERROR) << "SeConstArgs failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int PowerOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  int arg_cn = 0;
  if (broadcast_) {
    if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_tensors_.at(0)->data()) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  } else {
    if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_tensors_.at(0)->data()) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
    if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_tensors_.at(1)->data()) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_tensors_.at(0)->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->RunKernel(kernel_, global_range_, local_range_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_PowFusion, OpenCLKernelCreator<PowerOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_PowFusion, OpenCLKernelCreator<PowerOpenCLKernel>)
}  // namespace mindspore::kernel
