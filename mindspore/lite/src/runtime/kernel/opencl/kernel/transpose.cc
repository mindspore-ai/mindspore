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
#include "src/runtime/kernel/opencl/kernel/transpose.h"
#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/transpose.cl.inc"
#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Nchw2Nhwc;
using mindspore::schema::PrimitiveType_Nhwc2Nchw;
using mindspore::schema::PrimitiveType_Transpose;

namespace mindspore::kernel {

int TransposeOpenCLKernel::CheckSpecs() {
  auto param = reinterpret_cast<TransposeParameter *>(op_parameter_);
  if (in_tensors_[0]->shape().size() != 4 || in_tensors_[0]->shape()[0] > 1) {
    MS_LOG(ERROR) << "Transpose only support 4d tensor and n = 1 yet.";
    return mindspore::lite::RET_ERROR;
  }
  if (param->num_axes_ == 4 && param->perm_[0] == 0 && param->perm_[1] == 3 && param->perm_[2] == 1 &&
      param->perm_[3] == 2) {
    type = TransposeType::AXIS0312;
  } else if (param->num_axes_ == 4 && param->perm_[0] == 0 && param->perm_[1] == 2 && param->perm_[2] == 3 &&
             param->perm_[3] == 1) {
    type = TransposeType::AXIS0231;
  } else {
    MS_LOG(ERROR) << "unsupported transpose axes.";
    return mindspore::lite::RET_ERROR;
  }
  return RET_OK;
}

int TransposeOpenCLKernel::Prepare() {
  std::string kernel_name = "transpose";
  if (type == TransposeType::AXIS0312) {
    kernel_name += "_0312";
  } else if (type == TransposeType::AXIS0231) {
    kernel_name += "_0231";
  }
  if (in_tensors_[0]->shape()[2] * UP_DIV(in_tensors_[0]->shape()[3], C4NUM) > MAX_IMAGE2D_SIZE) {
    // just for input
    kernel_name += "_oversize";
  }
  kernel_name += "_NHWC4";

#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else
  std::set<std::string> build_options;
  std::string source = transpose_source;
  std::string program_name = "transpose";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif
  SetConstArgs();
  SetGlobalLocal();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return mindspore::lite::RET_OK;
}

void TransposeOpenCLKernel::SetConstArgs() {
  std::vector<int> shapex = out_tensors_[0]->shape();
  size_t n = shapex[0];  // n=1
  size_t h = shapex[1];
  size_t w = shapex[2];
  size_t c = shapex[3];
  int arg_idx = 2;
  cl_int4 shape = {static_cast<int>(n), static_cast<int>(h), static_cast<int>(w), static_cast<int>(c)};
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, shape);
}

void TransposeOpenCLKernel::SetGlobalLocal() {
  std::vector<int> shapex = out_tensors_[0]->shape();
  size_t h = shapex[1];
  size_t w = shapex[2];
  size_t c = shapex[3];
  size_t c4 = UP_DIV(c, 4);
  if (type == TransposeType::AXIS0312) {  // NHWC -> NCHW
    global_range_ = {UP_DIV(h, C4NUM), w, c4};
  } else if (type == TransposeType::AXIS0231) {  // NCHW -> NHWC
    global_range_ = {h, UP_DIV(w, C4NUM), c4};
  }
}

int TransposeOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  int arg_idx = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->data_c());
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr);
  return mindspore::lite::RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Transpose, OpenCLKernelCreator<TransposeOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Transpose, OpenCLKernelCreator<TransposeOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Nhwc2Nchw, OpenCLKernelCreator<TransposeOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Nhwc2Nchw, OpenCLKernelCreator<TransposeOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Nchw2Nhwc, OpenCLKernelCreator<TransposeOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Nchw2Nhwc, OpenCLKernelCreator<TransposeOpenCLKernel>)
}  // namespace mindspore::kernel
