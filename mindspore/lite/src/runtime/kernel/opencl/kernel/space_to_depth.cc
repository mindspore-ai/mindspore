/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <string>
#include "include/errorcode.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/kernel/space_to_depth.h"
#include "src/runtime/kernel/opencl/cl/space_to_depth.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::PrimitiveType_DepthToSpace;
using mindspore::schema::PrimitiveType_SpaceToDepth;

namespace mindspore::kernel {
int SpaceToDepthOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_1 || out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int SpaceToDepthOpenCLKernel::Prepare() {
  std::string kernel_name;
  in_shape_ = GpuTensorInfo(in_tensors_[0]);
  out_shape_ = GpuTensorInfo(out_tensors_[0]);
  if (type() == PrimitiveType_DepthToSpace) {
    kernel_name = "DepthToSpace";
  } else {
    kernel_name = "SpaceToDepth";
  }
  if (in_shape_.C % C4NUM == 0 && out_shape_.C % C4NUM == 0) {
    kernel_name += "Align";
  }
  std::string source = space_to_depth_source;
  const std::string program_name = "SpaceToDepth";
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
  if (SetConstArgs() != RET_OK) {
    MS_LOG(ERROR) << "SeConstArgs failed.";
    return RET_ERROR;
  }
  SetGlobalLocal();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}
int SpaceToDepthOpenCLKernel::SetConstArgs() {
  cl_int4 cl_in_shape = {static_cast<cl_int>(in_shape_.N), static_cast<cl_int>(in_shape_.H),
                         static_cast<cl_int>(in_shape_.W), static_cast<cl_int>(in_shape_.Slice)};
  cl_int4 cl_out_shape = {static_cast<cl_int>(out_shape_.N), static_cast<cl_int>(out_shape_.H),
                          static_cast<cl_int>(out_shape_.W), static_cast<cl_int>(out_shape_.Slice)};
  auto param = reinterpret_cast<SpaceToDepthParameter *>(op_parameter_);
  int arg_idx = 2;
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, cl_in_shape) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, cl_out_shape) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, param->block_size_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (type() == PrimitiveType_DepthToSpace) {
    int co_size = out_shape_.C;
    if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, co_size) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  } else {
    int ci_size = in_shape_.C;
    if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, ci_size) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
void SpaceToDepthOpenCLKernel::SetGlobalLocal() {
  local_size_ = {};
  global_size_ = {out_shape_.Slice, out_shape_.W, out_shape_.H * out_shape_.N};
  AlignGlobalLocal(global_size_, local_size_);
}

int SpaceToDepthOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  int arg_idx = 0;
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_SpaceToDepth, OpenCLKernelCreator<SpaceToDepthOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_SpaceToDepth, OpenCLKernelCreator<SpaceToDepthOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_DepthToSpace, OpenCLKernelCreator<SpaceToDepthOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_DepthToSpace, OpenCLKernelCreator<SpaceToDepthOpenCLKernel>)
}  // namespace mindspore::kernel
