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

#include <string>
#include <set>
#include <algorithm>
#include "src/common/utils.h"
#include "src/runtime/kernel/opencl/kernel/pad.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/kernel/opencl/cl/pad.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PaddingMode_CONSTANT;
using mindspore::schema::PrimitiveType_Pad;

namespace mindspore::kernel {

int PadOpenCLKernel::Init() {
  auto param = reinterpret_cast<PadParameter *>(op_parameter_);
  std::set<std::string> build_options;

  if (in_tensors_.empty()) {
    MS_LOG(ERROR) << "PadOpenCLKernel in_tensors is empty";
    return RET_ERROR;
  }
  if (out_tensors_.empty()) {
    MS_LOG(ERROR) << "PadOpenCLKernel out_tensors is empty";
    return RET_ERROR;
  }
  if (param->paddings_[0] || param->paddings_[1] || param->paddings_[6] || param->paddings_[7]) {
    MS_LOG(ERROR) << "PadOpenCLKernel not support pad at Batch/Channel axis";
    return RET_ERROR;
  }
  if (param->pad_mode_ != PaddingMode_CONSTANT) {
    MS_LOG(ERROR) << "PadOpenCLKernel only support CONSTANT MODE";
    return RET_ERROR;
  }

  auto input_tensor = in_tensors_[0];
  auto output_tensor = out_tensors_[0];

  CI_ = input_tensor->Channel();
  IH_ = input_tensor->Height();
  IW_ = input_tensor->Width();
  CO_ = output_tensor->Channel();
  OH_ = output_tensor->Height();
  OW_ = output_tensor->Width();
  CI_SLICES_ = UP_DIV(CI_, C4NUM);
  CO_SLICES_ = UP_DIV(CO_, C4NUM);

  const std::string source = pad_source;
  const std::string program_name = "Pad";
  const std::string kernel_name = "Pad_NHWC4";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);

  MS_LOG(DEBUG) << "Pad Init Done!";
  return RET_OK;
}

int PadOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";

  auto param = reinterpret_cast<PadParameter *>(op_parameter_);
  cl_int4 input_shape = {1, IH_, IW_, CI_SLICES_};
  cl_int4 output_shape = {1, OH_, OW_, CO_SLICES_};
  cl_int2 pad_top_left = {param->paddings_[2], param->paddings_[4]};

  int arg_cn = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_tensors_[0]->data_c(), lite::opencl::MemType::IMG);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_tensors_[0]->data_c(), lite::opencl::MemType::IMG);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, input_shape);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, output_shape);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, pad_top_left);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, static_cast<cl_float>(param->constant_value_));

  std::vector<size_t> global = {static_cast<size_t>(OH_), static_cast<size_t>(OW_), static_cast<size_t>(CO_SLICES_)};
  std::vector<size_t> local = {8, 4, 1};
  ocl_runtime_->RunKernel(kernel_, global, local, nullptr);

  return RET_OK;
}

kernel::LiteKernel *OpenCLPadKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                           const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                           const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                           const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) PadOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create OpenCL Pad kernel failed!";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: Pad";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Pad, OpenCLPadKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Pad, OpenCLPadKernelCreator)
}  // namespace mindspore::kernel
