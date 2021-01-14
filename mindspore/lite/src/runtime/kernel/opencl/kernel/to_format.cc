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

#include "src/runtime/kernel/opencl/kernel/to_format.h"
#include <set>
#include <map>
#include <string>
#include <utility>
#include "include/errorcode.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/cl/to_format.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::opencl::MemType;
using mindspore::schema::PrimitiveType_ToFormat;

namespace mindspore::kernel {

int ToFormatOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() != 1 || out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  auto data_type = in_tensors_.front()->data_type();
  if (data_type != kNumberTypeFloat32 && data_type != kNumberTypeFloat16 && data_type != kNumberTypeInt32) {
    MS_LOG(ERROR) << "Unsupported data type " << data_type;
    return RET_ERROR;
  }
  auto parameter = reinterpret_cast<OpenCLToFormatParameter *>(op_parameter_);
  out_mem_type_ = parameter->out_mem_type;
  return RET_OK;
}

void ToFormatOpenCLKernel::SetConstArgs() {
  cl_int4 shape{(cl_int)N_, (cl_int)H_, (cl_int)W_, (cl_int)C_};
  cl_int4 gsize{(cl_int)(N_ * H_), (cl_int)W_, (cl_int)UP_DIV(C_, C4NUM), 1};
  ocl_runtime_->SetKernelArg(kernel_, 2, gsize);
  ocl_runtime_->SetKernelArg(kernel_, 3, shape);
}

void ToFormatOpenCLKernel::SetGlobalLocal() {
  global_size_ = {N_ * H_, W_, UP_DIV(C_, C4NUM)};
  local_size_ = {8, 16, 3};
  size_t max_work_group_size = ocl_runtime_->DeviceMaxWorkGroupSize();
  if (max_work_group_size < 384) {
    local_size_[2] = 1;
  }
  OpenCLKernel::AlignGlobalLocal(global_size_, local_size_);
}

int ToFormatOpenCLKernel::Prepare() {
  std::map<TypeId, std::string> dtype_str{
    {kNumberTypeFloat32, "float"}, {kNumberTypeFloat16, "half"}, {kNumberTypeInt32, "float"}};
  std::string kernel_name;
  if (out_mem_type_ == MemType::IMG) {
    kernel_name = "to_format_NHWC_to_NHWC4_IMG_" + dtype_str[in_tensors_.front()->data_type()];
  } else {
    kernel_name = "to_format_NHWC4_to_NHWC_BUF_" + dtype_str[out_tensors_.front()->data_type()];
  }
  this->set_name(kernel_name);

#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else
  std::string program_name = "to_format";
  std::string source = to_format_source;
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);
#endif

  auto output = GpuTensorInfo(out_tensors_.front());
  N_ = output.N;
  H_ = output.H;
  W_ = output.W;
  C_ = output.C;

  SetGlobalLocal();
  SetConstArgs();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int ToFormatOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  auto src_mem_type = (out_mem_type_ == MemType::IMG) ? lite::opencl::MemType::BUF : lite::opencl::MemType::IMG;
  auto dst_mem_type = out_mem_type_;
  ocl_runtime_->SetKernelArg(kernel_, 0, in_tensors_.front()->data_c(), src_mem_type);
  ocl_runtime_->SetKernelArg(kernel_, 1, out_tensors_.front()->data_c(), dst_mem_type);
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return RET_OK;
}

int ToFormatOpenCLKernel::InferShape() {
  if (infer_shape_flag_) {
    return RET_OK;
  }
  if (in_tensors_[0]->shape().size() == 0 || in_tensors_[0]->ElementsNum() < 0) {
    MS_LOG(ERROR) << "to_format op in tensor shape is 0, infer shape failed!";
    return RET_ERROR;
  }
  out_tensors_[0]->set_shape(in_tensors_[0]->shape());
  infer_shape_flag_ = true;
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_ToFormat, OpenCLKernelCreator<ToFormatOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_ToFormat, OpenCLKernelCreator<ToFormatOpenCLKernel>)
}  // namespace mindspore::kernel
