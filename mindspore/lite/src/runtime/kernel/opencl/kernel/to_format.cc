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
  static std::map<TypeId, std::string> dtype_str{{kNumberTypeFloat32, "float32"},
                                                 {kNumberTypeFloat16, "float16"},
                                                 {kNumberTypeInt32, "int32"},
                                                 {kNumberTypeUInt32, "uint32"}};
  auto in_tensor = in_tensors_.front();
  auto out_tensor = out_tensors_.front();
  std::string kernel_name = out_mem_type_ == MemType::IMG ? "BUF_to_IMG_" : "IMG_to_BUF_";
  kernel_name += dtype_str[in_tensor->data_type()] + "_" + dtype_str[out_tensor->data_type()];
  this->set_name(kernel_name);

  std::string program_name = "to_format";
  std::string source = to_format_source;
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);

  auto output = GpuTensorInfo(out_tensor);
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
  if (!op_parameter_->infer_flag_) {
    op_parameter_->infer_flag_ = true;
    out_tensors_.front()->set_shape(in_tensors_.front()->shape());
  }
  return RET_OK;
}

}  // namespace mindspore::kernel
