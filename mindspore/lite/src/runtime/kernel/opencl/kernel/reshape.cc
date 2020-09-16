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

#include <set>
#include <string>
#include "include/errorcode.h"
#include "src/kernel_registry.h"
#include "src/runtime/opencl/opencl_runtime.h"
#include "src/runtime/kernel/opencl/kernel/reshape.h"
#include "src/runtime/kernel/opencl/cl/reshape.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Reshape;
using mindspore::schema::PrimitiveType_Squeeze;

namespace mindspore::kernel {

int ReshapeOpenCLKernel::Init() {
  std::string kernel_name = "reshape";
  kernel_name += "_" + std::string(EnumNameFormat(op_format_));
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  enable_fp16_ = ocl_runtime->GetFp16Enable();

  if (in_tensors_[0]->shape().back() != out_tensors_[0]->shape().back()) {
    MS_LOG(ERROR) << "Reshape input channel " << in_tensors_[0]->shape().back() << " should equal output channel"
                  << out_tensors_[0]->shape().back();
    return RET_ERROR;
  }
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime->GetKernelFromBinary(kernel_name);
#else
  std::set<std::string> build_options;
  std::string source = reshape_source;
  std::string program_name = "reshape";
  ocl_runtime->LoadSource(program_name, source);
  ocl_runtime->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif
  in_ori_format_ = in_tensors_[0]->GetFormat();
  out_ori_format_ = out_tensors_[0]->GetFormat();
  in_tensors_[0]->SetFormat(op_format_);
  out_tensors_[0]->SetFormat(op_format_);
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int ReshapeOpenCLKernel::ReSize() { return RET_OK; }

int ReshapeOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t im_dst_x, im_dst_y;
  std::vector<int> shapex = out_tensors_[0]->shape();
  int n, h, w, c;
  if (shapex.size() == 2) {
    n = shapex[0];
    h = w = 1;
    c = shapex[1];
  } else {
    n = shapex[0];
    h = shapex[1];
    w = shapex[2];
    c = shapex[3];
  }
  if (op_format_ == schema::Format::Format_NHWC4) {
    im_dst_x = w * UP_DIV(c, C4NUM);
    im_dst_y = n * h;
  } else if (op_format_ == schema::Format::Format_NC4HW4) {
    im_dst_x = w;
    im_dst_y = n * UP_DIV(c, C4NUM) * h;
  } else {
    MS_LOG(ERROR) << "not support op format:" << EnumNameFormat(op_format_);
    return RET_ERROR;
  }
  size_t img_dtype = CL_FLOAT;
  if (enable_fp16_) {
    img_dtype = CL_HALF_FLOAT;
  }
  img_size->clear();
  std::vector<size_t> vec{im_dst_x, im_dst_y, img_dtype};
  *img_size = vec;
  return RET_OK;
}

int ReshapeOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  std::vector<int> shapex = in_tensors_[0]->shape();
  int h = shapex[1];
  int w = shapex[2];
  int c = shapex[3];
  int c4 = UP_DIV(c, C4NUM);
  int oh, ow;
  if (out_tensors_[0]->shape().size() == 2) {
    oh = ow = 1;
  } else {
    oh = out_tensors_[0]->shape()[1];
    ow = out_tensors_[0]->shape()[2];
  }
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  std::vector<size_t> local = {};
  std::vector<size_t> global = {(size_t)oh, (size_t)ow, (size_t)c4};
  cl_int4 size = {h, w, c4, 1};
  cl_int4 size_out = {oh, ow, c4, 1};
  int arg_idx = 0;
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->MutableData());
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->MutableData());
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, size);
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, size_out);
  ocl_runtime->RunKernel(kernel_, global, local, nullptr);
  return RET_OK;
}

kernel::LiteKernel *OpenCLReshapeKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                               const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                               const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) ReshapeOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
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

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Reshape, OpenCLReshapeKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Reshape, OpenCLReshapeKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Squeeze, OpenCLReshapeKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Squeeze, OpenCLReshapeKernelCreator)
}  // namespace mindspore::kernel
