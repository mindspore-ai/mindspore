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

#include "src/runtime/kernel/opencl/kernel/pooling2d.h"
#include <string>
#include <set>
#include "include/errorcode.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/utils.h"
#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/pooling2d.cl.inc"
#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INVALID_OP_NAME;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;
using mindspore::lite::opencl::MemType;
using mindspore::schema::PrimitiveType_AvgPoolFusion;
using mindspore::schema::PrimitiveType_MaxPoolFusion;

namespace mindspore {
namespace kernel {

int PoolingOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() != 1 || out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  if (in_tensors_[0]->shape().size() != 4) {
    MS_LOG(ERROR) << "Only support 4d tensor.";
    return RET_ERROR;
  }
  if (parameter_->pool_mode_ != PoolMode_MaxPool && parameter_->pool_mode_ != PoolMode_AvgPool) {
    MS_LOG(ERROR) << "Init `Pooling2d` kernel failed, unsupported pool mode!";
    return RET_ERROR;
  }
  if (parameter_->act_type_ != ActType_No && parameter_->act_type_ != ActType_Relu) {
    MS_LOG(ERROR) << "Unsupported activation type " << parameter_->act_type_;
    return RET_ERROR;
  }
  return RET_OK;
}

int PoolingOpenCLKernel::Prepare() {
  std::string kernel_name;
  if (parameter_->pool_mode_ == PoolMode_MaxPool) {
    kernel_name = "MaxPooling2d";
  } else if (parameter_->pool_mode_ == PoolMode_AvgPool) {
    kernel_name = "AvgPooling2d";
  }
  switch (parameter_->act_type_) {
    case ActType_No:
      break;
    case ActType_Relu:
      kernel_name += "_ReLU";
      break;
    default:
      MS_LOG(ERROR) << "Unsupported activation type " << parameter_->act_type_;
      break;
  }
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else
  kernel_name += "_NHWC4";
  kernel_name += "_IMG";
  std::string source = pooling2d_source;
  std::string program_name = "Pooling2d";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);
#endif
  SetConstArgs();
  SetGlobalLocal();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";

  return mindspore::lite::RET_OK;
}

void PoolingOpenCLKernel::SetGlobalLocal() {
  const size_t global_x = out_tensors_[0]->shape()[1] * out_tensors_[0]->shape()[0];
  const size_t global_y = out_tensors_[0]->shape()[2];
  const size_t global_z = UP_DIV(out_tensors_[0]->shape()[3], C4NUM);
  global_size_ = {global_z, global_y, global_x};
  local_size_ = {};
  AlignGlobalLocal(global_size_, local_size_);
}

void PoolingOpenCLKernel::SetConstArgs() {
  int slices = UP_DIV(out_tensors_[0]->shape()[3], C4NUM);
  cl_int4 input_shape = {in_tensors_[0]->shape()[0], in_tensors_[0]->shape()[1], in_tensors_[0]->shape()[2], slices};
  cl_int4 output_shape = {out_tensors_[0]->shape()[0], out_tensors_[0]->shape()[1], out_tensors_[0]->shape()[2],
                          slices};
  cl_int2 stride = {parameter_->stride_h_, parameter_->stride_w_};
  cl_int2 kernel_size = {parameter_->window_h_, parameter_->window_w_};
  cl_int2 padding = {parameter_->pad_u_, parameter_->pad_l_};
  int arg_idx = 2;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, input_shape);
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, output_shape);
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, stride);
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, kernel_size);
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, padding);
}

int PoolingOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  int arg_idx = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->data_c());
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return mindspore::lite::RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_AvgPoolFusion, OpenCLKernelCreator<PoolingOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_MaxPoolFusion, OpenCLKernelCreator<PoolingOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_AvgPoolFusion, OpenCLKernelCreator<PoolingOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_MaxPoolFusion, OpenCLKernelCreator<PoolingOpenCLKernel>)
}  // namespace kernel
}  // namespace mindspore
