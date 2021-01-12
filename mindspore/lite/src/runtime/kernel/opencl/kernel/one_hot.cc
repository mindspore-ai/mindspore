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

#include <set>
#include <string>
#include <map>
#include "include/errorcode.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/kernel/one_hot.h"
#include "src/runtime/kernel/opencl/cl/one_hot.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_OneHot;

namespace mindspore::kernel {
int OneHotOpenCLKernel::CheckSpecs() {
  if ((in_tensors_.size() < 2 || in_tensors_.size() > 4) || out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int OneHotOpenCLKernel::Prepare() {
  std::string kernel_name = "OneHot";
  auto param = reinterpret_cast<OneHotParameter *>(op_parameter_);
  in_shape_ = GpuTensorInfo(in_tensors_[0]);
  out_shape_ = GpuTensorInfo(out_tensors_[0]);
  axis_ = out_shape_.AlignAxis(param->axis_);
  if (in_tensors_[0]->shape().size() == 1 && axis_ == 0) {
    kernel_name += "2DAxis0";
  } else {
    kernel_name += "Axis" + std::to_string(axis_);
  }
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else
  std::string source = one_hot_source;
  std::string program_name = "OneHot";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);
#endif
  InitWeights();
  SetConstArgs();
  SetGlobalLocal();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return mindspore::lite::RET_OK;
}

int OneHotOpenCLKernel::InitWeights() {
  depth_ = static_cast<int32_t *>(in_tensors_[1]->data_c())[0];
  if (in_tensors_.size() > 2) {
    on_value_ = static_cast<float *>(in_tensors_[2]->data_c())[0];
  }
  if (in_tensors_.size() > 3) {
    off_value_ = static_cast<float *>(in_tensors_[3]->data_c())[0];
  }
  return RET_OK;
}

void OneHotOpenCLKernel::SetConstArgs() {
  cl_int2 cl_in_image2d_shape = {static_cast<cl_int>(in_shape_.width), static_cast<cl_int>(in_shape_.height)};
  cl_int4 cl_out_shape = {static_cast<cl_int>(out_shape_.N), static_cast<cl_int>(out_shape_.H),
                          static_cast<cl_int>(out_shape_.W), static_cast<cl_int>(out_shape_.Slice)};
  int arg_idx = 2;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, cl_in_image2d_shape);
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, cl_out_shape);
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, depth_);
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, on_value_);
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, off_value_);
  ocl_runtime_->SetKernelArg(kernel_, arg_idx, static_cast<int>(out_shape_.C));
}
void OneHotOpenCLKernel::SetGlobalLocal() {
  local_size_ = {};
  global_size_ = {out_shape_.Slice, out_shape_.W, out_shape_.H * out_shape_.N};
  AlignGlobalLocal(global_size_, local_size_);
}

int OneHotOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  ocl_runtime_->SetKernelArg(kernel_, 0, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, 1, out_tensors_[0]->data_c());
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return mindspore::lite::RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_OneHot, OpenCLKernelCreator<OneHotOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_OneHot, OpenCLKernelCreator<OneHotOpenCLKernel>)
}  // namespace mindspore::kernel
