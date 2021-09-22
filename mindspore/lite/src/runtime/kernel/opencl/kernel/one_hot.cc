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

#include <string>
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
  if ((in_tensors_.size() < INPUT_TENSOR_SIZE_2 || in_tensors_.size() > INPUT_TENSOR_SIZE_4) ||
      out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int OneHotOpenCLKernel::Prepare() {
  std::string kernel_name = "OneHot";
  param_ = reinterpret_cast<OneHotParameter *>(op_parameter_);
  in_shape_ = GpuTensorInfo(in_tensors_[0]);
  out_shape_ = GpuTensorInfo(out_tensors_[0]);
  axis_ = out_shape_.AlignAxis(param_->axis_);
  if (in_tensors_[0]->shape().size() == DIMENSION_1D && axis_ == 3) {
    kernel_name += "2DAxis3";
  } else {
    kernel_name += "Axis" + std::to_string(axis_);
  }
  std::string source = one_hot_source;
  const std::string program_name = "OneHot";
  if (!ocl_runtime_->LoadSource(program_name, source)) {
    MS_LOG(ERROR) << "Load source failed.";
    return RET_ERROR;
  }
  std::vector<std::string> build_options_ext;
  if (ocl_runtime_->GetFp16Enable()) {
    build_options_ext = {" -DWRITE_IMAGE=write_imageh -DREAD_IMAGE=read_imagei "};
  } else {
    build_options_ext = {" -DWRITE_IMAGE=write_imagef -DREAD_IMAGE=read_imagei "};
  }
  auto ret = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options_ext);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build kernel failed.";
    return ret;
  }
  InitWeights();
  if (SetConstArgs() != RET_OK) {
    MS_LOG(ERROR) << "SeConstArgs failed.";
    return RET_ERROR;
  }
  SetGlobalLocal();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int OneHotOpenCLKernel::InitWeights() {
  depth_ = static_cast<int32_t *>(in_tensors_[1]->data())[0];
  MS_ASSERT(depth_);
  // inputs num is 3 or 4.
  if (in_tensors_.size() == INPUT_TENSOR_SIZE_3) {  // onnx
    off_value_ = static_cast<float *>(in_tensors_[2]->data())[0];
    on_value_ = static_cast<float *>(in_tensors_[2]->data())[1];
    param_->support_neg_index_ = true;
  }
  if (in_tensors_.size() == INPUT_TENSOR_SIZE_4) {  // tf
    on_value_ = static_cast<float *>(in_tensors_[2]->data())[0];
    off_value_ = static_cast<float *>(in_tensors_[3]->data())[0];
    param_->support_neg_index_ = false;
  }
  MS_ASSERT(off_value_);
  MS_ASSERT(on_value_);
  return RET_OK;
}

int OneHotOpenCLKernel::SetConstArgs() {
  cl_int2 cl_in_image2d_shape = {static_cast<cl_int>(in_shape_.width), static_cast<cl_int>(in_shape_.height)};
  cl_int4 cl_out_shape = {static_cast<cl_int>(out_shape_.N), static_cast<cl_int>(out_shape_.H),
                          static_cast<cl_int>(out_shape_.W), static_cast<cl_int>(out_shape_.Slice)};
  int arg_idx = 2;
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, cl_in_image2d_shape) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, cl_out_shape) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, depth_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, on_value_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, off_value_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, static_cast<int>(out_shape_.C)) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx, static_cast<int>(param_->support_neg_index_)) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
void OneHotOpenCLKernel::SetGlobalLocal() {
  local_size_ = {};
  global_size_ = {out_shape_.Slice, out_shape_.W, out_shape_.H * out_shape_.N};
  AlignGlobalLocal(global_size_, local_size_);
}

int OneHotOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  if (ocl_runtime_->SetKernelArg(kernel_, 0, in_tensors_[0]->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, 1, out_tensors_[0]->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeInt32, PrimitiveType_OneHot, OpenCLKernelCreator<OneHotOpenCLKernel>)
}  // namespace mindspore::kernel
