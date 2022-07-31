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

#include "src/litert/kernel/opencl/kernel/to_format.h"
#include <map>
#include <string>
#include "include/errorcode.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/opencl/cl/to_format.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::opencl::MemType;

namespace mindspore::kernel {
int ToFormatOpenCLKernel::CheckSpecsWithoutShape() {
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_1 || out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  auto data_type = in_tensors_.front()->data_type();
  if (data_type != kNumberTypeFloat32 && data_type != kNumberTypeFloat16 && data_type != kNumberTypeInt32 &&
      data_type != kNumberTypeInt8) {
    MS_LOG(WARNING) << "Unsupported data type " << data_type;
    return RET_ERROR;
  }
  return RET_OK;
}

int ToFormatOpenCLKernel::CheckSpecs() { return RET_OK; }

int ToFormatOpenCLKernel::SetConstArgs() {
  cl_int4 shape{(cl_int)N_, (cl_int)(H_ * D_), (cl_int)W_, (cl_int)C_};
  cl_int4 gsize{(cl_int)(N_ * D_ * H_), (cl_int)W_, (cl_int)UP_DIV(C_, C4NUM), 1};
  if (ocl_runtime_->SetKernelArg(kernel_, CLARGSINDEX2, gsize) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, CLARGSINDEX3, shape) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ToFormatOpenCLKernel::SetGlobalLocal() {
  global_size_ = {N_ * D_ * H_, W_, UP_DIV(C_, C4NUM)};
  local_size_ = {8, 16, 3};  // local_x : 3, local_y : 16, local_z : 3
  size_t max_work_group_size = ocl_runtime_->DeviceMaxWorkGroupSize();
  if (max_work_group_size < 384) {  // max work group size : 384
    local_size_[CLIDX_Z] = 1;
  }
  OpenCLKernel::AlignGlobalLocal(global_size_, local_size_);

  return RET_OK;
}

int ToFormatOpenCLKernel::Prepare() {
  static std::map<TypeId, std::string> dtype_str{
    {kNumberTypeFloat32, "float32"}, {kNumberTypeFloat16, "float16"}, {kNumberTypeInt32, "int32"},
    {kNumberTypeUInt32, "uint32"},   {kNumberTypeInt8, "int8"},
  };
  auto in_tensor = in_tensors_.front();
  auto out_tensor = out_tensors_.front();
  std::string kernel_name = out_mem_type_ == MemType::IMG ? "BUF_to_IMG_" : "IMG_to_BUF_";
  kernel_name += dtype_str[in_tensor->data_type()] + "_" + dtype_str[out_tensor->data_type()];
  this->set_name(kernel_name);

  const std::string program_name = "to_format";
  std::string source = to_format_source;
  if (!ocl_runtime_->LoadSource(program_name, source)) {
    MS_LOG(ERROR) << "Load source failed.";
    return RET_ERROR;
  }
  auto ret = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build kernel failed.";
    return ret;
  }

  auto output = GpuTensorInfo::CreateGpuTensorInfo(out_tensor);
  N_ = output->N;
  D_ = output->D;
  H_ = output->H;
  W_ = output->W;
  C_ = output->C;

  (void)SetGlobalLocal();
  if (SetConstArgs() != RET_OK) {
    MS_LOG(ERROR) << "SeConstArgs failed.";
    return RET_ERROR;
  }
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int ToFormatOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  auto src_mem_type = (out_mem_type_ == MemType::IMG) ? lite::opencl::MemType::BUF : lite::opencl::MemType::IMG;
  auto dst_mem_type = out_mem_type_;
  if (ocl_runtime_->SetKernelArg(kernel_, CLARGSINDEX0, in_tensors_.front()->data(),
                                 (src_mem_type == lite::opencl::MemType::BUF)) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, CLARGSINDEX1, out_tensors_.front()->data(),
                                 (dst_mem_type == lite::opencl::MemType::BUF)) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ToFormatOpenCLKernel::InferShape() {
  if (!InferShapeDone()) {
    out_tensors_.front()->set_shape(in_tensors_.front()->shape());
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
