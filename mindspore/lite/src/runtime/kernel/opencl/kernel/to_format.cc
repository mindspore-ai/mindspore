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
using mindspore::schema::PrimitiveType_ToFormat;

namespace mindspore::kernel {

int ToFormatOpenCLKernel::Init() {
  auto parameter = reinterpret_cast<OpenCLToFormatParameter *>(op_parameter_);
  out_mem_type_ = parameter->out_mem_type;
  std::map<TypeId, std::string> dtype_str{{kNumberTypeFloat32, "float"}, {kNumberTypeFloat16, "half"}};
  std::string kernel_name;
  if (parameter->out_mem_type == OpenCLMemType::IMG) {
    kernel_name = "to_format_NHWC_to_NHWC4_IMG_" + dtype_str[in_tensors_[0]->data_type()];
  } else {
    kernel_name = "to_format_NHWC4_to_NHWC_BUF_" + dtype_str[out_tensors_[0]->data_type()];
  }
  this->set_name(kernel_name);

#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else
  std::string program_name = "to_format";
  std::set<std::string> build_options;
  std::string source = to_format_source;
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif

  InitNHWC();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int ToFormatOpenCLKernel::InitNHWC() {
  std::vector<int> out_shape = out_tensors_[0]->shape();
  if (out_shape.size() == 1) {
    N_ = out_shape[0];
    H_ = 1;
    W_ = 1;
    C_ = 1;
  } else if (out_shape.size() == 2) {
    N_ = out_shape[0];
    H_ = 1;
    W_ = 1;
    C_ = out_shape[1];
  } else if (out_shape.size() == 3) {
    N_ = out_shape[0];
    H_ = 1;
    W_ = out_shape[1];
    C_ = out_shape[2];
  } else if (out_shape.size() == 4) {
    N_ = out_shape[0];
    H_ = out_shape[1];
    W_ = out_shape[2];
    C_ = out_shape[3];
  }
  return RET_OK;
}

int ToFormatOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t img_height = N_ * H_;
  size_t img_width = W_ * UP_DIV(C_, C4NUM);
  size_t img_dtype = ocl_runtime_->GetFp16Enable() ? CL_HALF_FLOAT : CL_FLOAT;
  *img_size = {img_width, img_height, img_dtype};
  return RET_OK;
}

int ToFormatOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  std::vector<size_t> global = {N_ * H_, W_, UP_DIV(C_, C4NUM)};
  std::vector<size_t> local = {16, 8, 1};
  cl_int4 shape{(cl_int)N_, (cl_int)H_, (cl_int)W_, (cl_int)C_};
  cl_int4 gsize{(cl_int)global[0], (cl_int)global[1], (cl_int)global[2], 1};

  auto src_mem_type = (out_mem_type_ == OpenCLMemType::IMG) ? lite::opencl::MemType::BUF : lite::opencl::MemType::IMG;
  auto dst_mem_type = (out_mem_type_ == OpenCLMemType::IMG) ? lite::opencl::MemType::IMG : lite::opencl::MemType::BUF;
  ocl_runtime_->SetKernelArg(kernel_, 0, in_tensors_[0]->data_c(), src_mem_type);
  ocl_runtime_->SetKernelArg(kernel_, 1, out_tensors_[0]->data_c(), dst_mem_type);
  ocl_runtime_->SetKernelArg(kernel_, 2, gsize);
  ocl_runtime_->SetKernelArg(kernel_, 3, shape);
  ocl_runtime_->RunKernel(kernel_, global, local, nullptr);
  return RET_OK;
}

kernel::LiteKernel *OpenCLToFormatKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                                const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                                const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) ToFormatOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel " << opParameter->name_ << " create failed.";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_ToFormat, OpenCLToFormatKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_ToFormat, OpenCLToFormatKernelCreator)
}  // namespace mindspore::kernel
