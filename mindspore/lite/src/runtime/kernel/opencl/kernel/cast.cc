
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
#include "src/runtime/kernel/opencl/kernel/cast.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/kernel/opencl/cl/cast.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Cast;

namespace mindspore::kernel {

int CastOpenCLKernel::GetKernelName(std::string *kernel_name, CastParameter *param) {
  if (param->src_type_ == kNumberTypeFloat32 && param->dst_type_ == kNumberTypeFloat16) {
    kernel_name[0] += "_Fp32ToFp16";
  } else if (param->src_type_ == kNumberTypeFloat16 && param->dst_type_ == kNumberTypeFloat32) {
    kernel_name[0] += "_Fp16ToFp32";
  } else {
    MS_LOG(ERROR) << "unsupported convert format from : " << param->src_type_ << "to  " << param->dst_type_;
    return RET_ERROR;
  }
  return RET_OK;
}

int CastOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() != 1 || out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  if (in_tensors_.at(0)->shape().size() == 4) {
    MS_LOG(ERROR) << "The dim of in_tensors->shape must be 4 but your dim is : " << in_tensors_.at(0)->shape().size();
    return RET_ERROR;
  }
  return RET_OK;
}

void CastOpenCLKernel::SetConstArgs() {
  auto input_shape = in_tensors_[0]->shape();
  cl_int4 input_shape_ = {input_shape[0], input_shape[1], input_shape[2], UP_DIV(input_shape[3], C4NUM)};
  int arg_cn = 2;
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, input_shape_);
}

void CastGetWorkGroup(const std::vector<size_t> &global, std::vector<size_t> *local, int max_size) {
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

void CastOpenCLKernel::SetGlobalLocal() {
  auto input_shape = in_tensors_[0]->shape();
  uint32_t OH = input_shape[1];
  uint32_t OW = input_shape[2];
  uint32_t OC = UP_DIV(input_shape[3], C4NUM);

  const std::vector<size_t> &max_global = ocl_runtime_->GetWorkItemSize();
  local_size_ = {1, 1, 1};  // init local
  global_size_ = {OH, OW, OC};
  CastGetWorkGroup(global_size_, &local_size_, max_global[0]);
  OpenCLKernel::AlignGlobalLocal(global_size_, local_size_);
}

int CastOpenCLKernel::Prepare() {
  auto param = reinterpret_cast<CastParameter *>(this->op_parameter_);
  std::string kernel_name = "Cast";
  GetKernelName(&kernel_name, param);
  kernel_name += "_NHWC4";
  std::string source = cast_source;
  std::string program_name = "cast";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  SetConstArgs();
  SetGlobalLocal();
  return RET_OK;
}

int CastOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  int arg_cn = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_tensors_[0]->data_c());   // input tensor
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_tensors_[0]->data_c());  // out tensor
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Cast, OpenCLKernelCreator<CastOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Cast, OpenCLKernelCreator<CastOpenCLKernel>);
}  // namespace mindspore::kernel
