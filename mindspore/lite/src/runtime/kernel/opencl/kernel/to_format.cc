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
#include "src/runtime/opencl/opencl_runtime.h"
#include "src/runtime/kernel/opencl/cl/fp32/to_format.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ToFormat;

namespace mindspore::kernel {

int ToFormatOpenCLKernel::Init() {
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  auto parameter = reinterpret_cast<OpenCLToFormatParameter *>(op_parameter_);
  out_mem_type_ = parameter->out_mem_type;
  std::string program_name = "to_format";
  std::map<schema::Format, std::string> format_str{{schema::Format_NCHW, "NCHW"},
                                                   {schema::Format_NHWC, "NHWC"},
                                                   {schema::Format_NC4HW4, "NC4HW4"},
                                                   {schema::Format_NHWC4, "NHWC4"}};
  std::string kernel_name =
    "to_format_" + format_str[in_tensors_[0]->GetFormat()] + "_to_" + format_str[out_tensors_[0]->GetFormat()];
  if (out_mem_type_ == OpenCLMemType::IMG) {
    kernel_name += "_IMG";
  } else {
    kernel_name += "_BUF";
  }

#ifdef PROGRAM_WITH_IL
  ocl_runtime->CreateKernelFromIL(kernel_(), kernel_name);
#else
  std::set<std::string> build_options;
#ifdef ENABLE_FP16
  std::string source = to_format_source_fp16;
#else
  std::string source = to_format_source_fp32;
#endif
  ocl_runtime->LoadSource(program_name, source);
  ocl_runtime->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int ToFormatOpenCLKernel::ReSize() { return RET_OK; }

int ToFormatOpenCLKernel::GetGlobalSize(size_t idx, std::vector<size_t> *global_size) {
  std::vector<int> shapex = out_tensors_[0]->shape();
  if (out_tensors_[0]->GetFormat() == schema::Format_NHWC4 || out_tensors_[0]->GetFormat() == schema::Format_NHWC) {
    int h = shapex[1];
    int w = shapex[2];
    int c = shapex[3];
    int c4 = UP_DIV(c, C4NUM);
    std::vector<size_t> vec = {(size_t)h, (size_t)w, (size_t)c4};
    *global_size = std::move(vec);
  } else if (out_tensors_[0]->GetFormat() == schema::Format_NC4HW4 ||
             out_tensors_[0]->GetFormat() == schema::Format_NCHW) {
    int h = shapex[2];
    int w = shapex[3];
    int c = shapex[1];
    int c4 = UP_DIV(c, C4NUM);
    std::vector<size_t> vec = {(size_t)c4, (size_t)h, (size_t)w};
    *global_size = std::move(vec);
  } else if (out_tensors_[0]->GetFormat() == out_tensors_[0]->GetFormat() == schema::Format_NCHW) {
    int h = shapex[2];
    int w = shapex[3];
    int c = shapex[1];
    int w4 = UP_DIV(w, C4NUM);
    std::vector<size_t> vec = {(size_t)w4, (size_t)h, (size_t)c};
    *global_size = std::move(vec);
  }
  return RET_OK;
}
int ToFormatOpenCLKernel::GetLocalSize(size_t idx, const std::vector<size_t> &global_size,
                                       std::vector<size_t> *local_size) {
  return RET_OK;
}

int ToFormatOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t im_dst_x, im_dst_y;
  std::vector<int> shapex = out_tensors_[0]->shape();
  if (out_tensors_[0]->GetFormat() == schema::Format_NC4HW4) {
    int h = shapex[1];
    int w = shapex[2];
    int c = shapex[3];
    im_dst_y = UP_DIV(h * c, C4NUM);
    im_dst_x = w;
  } else if (out_tensors_[0]->GetFormat() == schema::Format_NHWC4) {
    int h = shapex[2];
    int w = shapex[3];
    int c = shapex[1];
    im_dst_x = UP_DIV(w * c, C4NUM);
    im_dst_y = h;
  } else {
    MS_LOG(ERROR) << "Unsupported format. " << out_tensors_[0]->GetFormat();
  }
  img_size->clear();
#ifdef ENABLE_FP16
  size_t img_dtype = CL_HALF_FLOAT;
#else
  size_t img_dtype = CL_FLOAT;
#endif
  std::vector<size_t> vec{im_dst_x, im_dst_y, img_dtype};
  *img_size = vec;
  return RET_OK;
}

int ToFormatOpenCLKernel::Run() {
  MS_LOG(DEBUG) << "ToFormat" << " Running!";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  std::vector<size_t> local = {};
  std::vector<size_t> global;
  GetGlobalSize(0, &global);
  auto shapex = in_tensors_[0]->shape();
  cl_int4 shape{shapex.size() > 0 ? shapex[0] : 1, shapex.size() > 1 ? shapex[1] : 1, shapex.size() > 2 ? shapex[2] : 1,
                shapex.size() > 3 ? shapex[3] : 1};
  cl_int4 gsize{(cl_int)global[0], (cl_int)global[1], (cl_int)global[2], 1};
  ocl_runtime->SetKernelArg(kernel_, 0, in_tensors_[0]->Data());
  ocl_runtime->SetKernelArg(kernel_, 1, out_tensors_[0]->Data());
  ocl_runtime->SetKernelArg(kernel_, 2, gsize);
  ocl_runtime->SetKernelArg(kernel_, 3, shape);
  ocl_runtime->RunKernel(kernel_, global, local, nullptr);
  return RET_OK;
}

kernel::LiteKernel *OpenCLToFormatKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                const std::vector<lite::tensor::Tensor *> &outputs,
                                                OpParameter *opParameter, const lite::Context *ctx,
                                                const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  auto *kernel = new (std::nothrow) ToFormatOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel " << opParameter->name_ << " create failed.";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_ToFormat, OpenCLToFormatKernelCreator)
}  // namespace mindspore::kernel
