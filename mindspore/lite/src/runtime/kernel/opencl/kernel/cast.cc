
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
#include <map>
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

int CastOpenCLKernel::CheckSpecs() {
  // the 2nd tensor is DstType
  if (in_tensors_.size() != 2 || out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  if (in_tensors_.front()->shape() != out_tensors_.front()->shape()) {
    MS_LOG(ERROR) << "input shape must be equal to output shape";
    return RET_ERROR;
  }
  auto input_dtype = in_tensors_.front()->data_type();
  if (input_dtype != kNumberTypeFloat32 && input_dtype != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "input dtype must be float32/float16";
    return RET_ERROR;
  }
  auto output_dtype = out_tensors_.front()->data_type();
  if (output_dtype != kNumberTypeFloat32 && output_dtype != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "output dtype must be float32/float16";
    return RET_ERROR;
  }
  return RET_OK;
}

void CastOpenCLKernel::SetConstArgs() {
  cl_int2 shape = {static_cast<int>(shape_.width), static_cast<int>(shape_.height)};
  ocl_runtime_->SetKernelArg(kernel_, 2, shape);
}

void CastOpenCLKernel::SetGlobalLocal() {
  global_size_ = {shape_.width, shape_.height};
  OpenCLKernel::AlignGlobalLocal(global_size_, {});
}

int CastOpenCLKernel::Prepare() {
  shape_ = GpuTensorInfo(in_tensors_.front());
  std::map<int, std::string> dtype_names = {
    {kNumberTypeFloat32, "fp32"},
    {kNumberTypeFloat16, "fp16"},
  };
  std::string program_name = "Cast";
  std::string kernel_name =
    "Cast_" + dtype_names[in_tensors_.front()->data_type()] + "_to_" + dtype_names[out_tensors_.front()->data_type()];
  ocl_runtime_->LoadSource(program_name, cast_source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);
  SetConstArgs();
  SetGlobalLocal();
  return RET_OK;
}

int CastOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  ocl_runtime_->SetKernelArg(kernel_, 0, in_tensors_.front()->data_c());
  ocl_runtime_->SetKernelArg(kernel_, 1, out_tensors_.front()->data_c());
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Cast, OpenCLKernelCreator<CastOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Cast, OpenCLKernelCreator<CastOpenCLKernel>);
}  // namespace mindspore::kernel
