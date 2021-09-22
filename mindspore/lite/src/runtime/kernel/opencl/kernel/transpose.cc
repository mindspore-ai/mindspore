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
#include "include/errorcode.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/kernel/transpose.h"
#include "src/runtime/kernel/opencl/cl/transpose.cl.inc"
#include "src/runtime/kernel/opencl/utils.h"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Transpose;

namespace mindspore::kernel {
int TransposeOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_2 || out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "Transpose input output size unsupported.";
    return RET_ERROR;
  }
  int in_ndim = in_tensors_.at(0)->shape().size();
  int out_ndim = out_tensors_.at(0)->shape().size();
  if (in_ndim != out_ndim) {
    MS_LOG(WARNING) << "Transpose only support in_ndim equal to out_ndim.";
    return RET_ERROR;
  }
  if (in_ndim > DIMENSION_4D) {
    MS_LOG(WARNING) << "Transpose don't support 5d tensor or higher.";
    return RET_ERROR;
  }
  if (CheckParamLikeTensor("Transpose", "perm", in_tensors_.at(1), kNumberTypeInt32, {in_ndim}) != RET_OK) {
    return RET_ERROR;
  }
  return RET_OK;
}

int TransposeOpenCLKernel::Prepare() {
  tensor_size_ = GpuTensorInfo(out_tensors_.front());
  auto *perm = reinterpret_cast<int32_t *>(in_tensors_.at(1)->data());
  int num_axes = in_tensors_.at(1)->shape().at(0);
  if (tensor_size_.NDim == DIMENSION_2D) {
    perm_4d_[0] = tensor_size_.AlignAxis(perm[0]);
    perm_4d_[1] = 1;
    perm_4d_[2] = 2;
    perm_4d_[3] = tensor_size_.AlignAxis(perm[1]);
    if (num_axes != tensor_size_.NDim) {
      perm_4d_[0] = 0;
      perm_4d_[1] = 1;
      perm_4d_[2] = 2;
      perm_4d_[3] = 3;
    }
  } else if (tensor_size_.NDim == DIMENSION_3D) {
    perm_4d_[0] = tensor_size_.AlignAxis(perm[0]);
    perm_4d_[1] = 1;
    perm_4d_[2] = tensor_size_.AlignAxis(perm[1]);
    perm_4d_[3] = tensor_size_.AlignAxis(perm[2]);
  } else if (tensor_size_.NDim == DIMENSION_4D) {
    perm_4d_[0] = tensor_size_.AlignAxis(perm[0]);
    perm_4d_[1] = tensor_size_.AlignAxis(perm[1]);
    perm_4d_[2] = tensor_size_.AlignAxis(perm[2]);
    perm_4d_[3] = tensor_size_.AlignAxis(perm[3]);
  } else {
    perm_4d_[0] = 0;
    perm_4d_[1] = 1;
    perm_4d_[2] = 2;
    perm_4d_[3] = 3;
  }
  std::string kernel_name = "transpose";

  if (tensor_size_.N == 1 && perm_4d_[0] == 0 && perm_4d_[1] == 3 && perm_4d_[2] == 1 && perm_4d_[3] == 2) {
    type_ = TransposeType::AXIS0312;
    kernel_name += "_0312";
  } else if (tensor_size_.N == 1 && perm_4d_[0] == 0 && perm_4d_[1] == 2 && perm_4d_[2] == 3 && perm_4d_[3] == 1) {
    type_ = TransposeType::AXIS0231;
    kernel_name += "_0231";
  } else {
    type_ = TransposeType::GENERAL;
    kernel_name += "_general";
  }

  if (in_tensors_[0]->shape().size() == DIMENSION_4D &&
      in_tensors_[0]->shape()[2] * UP_DIV(in_tensors_[0]->shape()[3], C4NUM) > ocl_runtime_->GetMaxImage2DWidth()) {
    // just for input
    kernel_name += "_oversize";
  }
  kernel_name += "_NHWC4";

  std::string source = transpose_source;
  const std::string program_name = "transpose";
  if (!ocl_runtime_->LoadSource(program_name, source)) {
    MS_LOG(ERROR) << "Load source failed.";
    return RET_ERROR;
  }
  auto build_options_ext = CreateBuildOptionsExtByDType(this->registry_data_type_);

  auto ret = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options_ext);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build kernel failed.";
    return ret;
  }
  if (SetConstArgs() != RET_OK) {
    MS_LOG(ERROR) << "SeConstArgs failed.";
    return RET_ERROR;
  }
  SetGlobalLocal();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int TransposeOpenCLKernel::SetConstArgs() {
  size_t n = tensor_size_.N;
  size_t h = tensor_size_.H;
  size_t w = tensor_size_.W;
  size_t c = tensor_size_.C;
  int arg_idx = 2;
  cl_int4 shape = {static_cast<int>(n), static_cast<int>(h), static_cast<int>(w), static_cast<int>(c)};
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, shape) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (type_ == TransposeType::GENERAL) {
    int de_perm[4];  // output to input perm
    for (int i = 0; i < 4; i++) {
      de_perm[perm_4d_[i]] = i;
    }
    cl_int4 de_perm_cl = {de_perm[0], de_perm[1], de_perm[2], de_perm[3]};
    if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, de_perm_cl) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
    GpuTensorInfo in_shape = GpuTensorInfo(in_tensors_[0]);
    cl_int4 in_shape_int4 = {static_cast<cl_int>(in_shape.N), static_cast<cl_int>(in_shape.H),
                             static_cast<cl_int>(in_shape.W), static_cast<cl_int>(in_shape.C)};
    if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_shape_int4) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

void TransposeOpenCLKernel::SetGlobalLocal() {
  size_t n = tensor_size_.N;
  size_t h = tensor_size_.H;
  size_t w = tensor_size_.W;
  size_t c = tensor_size_.C;
  size_t c4 = UP_DIV(c, 4);
  local_size_ = {};
  if (type_ == TransposeType::AXIS0312) {  // NHWC -> NCHW
    global_size_ = {UP_DIV(h, C4NUM), w, c4};
  } else if (type_ == TransposeType::AXIS0231) {  // NCHW -> NHWC
    global_size_ = {h, UP_DIV(w, C4NUM), c4};
  } else {  // general
    global_size_ = {n * h, w, c4};
  }
  AlignGlobalLocal(global_size_, local_size_);
}

int TransposeOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  int arg_idx = 0;
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Transpose, OpenCLKernelCreator<TransposeOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Transpose, OpenCLKernelCreator<TransposeOpenCLKernel>)
}  // namespace mindspore::kernel
