/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/opencl/kernel/transpose.h"
#include "src/litert/kernel/opencl/cl/transpose.cl.inc"
#include "src/litert/kernel/opencl/utils.h"

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

void TransposeOpenCLKernel::BroadCastPerm() {
  tensor_size_ = GpuTensorInfo(out_tensors_.front());
  auto *perm = reinterpret_cast<int32_t *>(in_tensors_.at(1)->data());
  int num_axes = in_tensors_.at(1)->shape().at(0);
  if (tensor_size_.NDim == DIMENSION_2D) {
    perm_4d_[kNHWC_N] = tensor_size_.AlignAxis(perm[kNHWC_N]);
    perm_4d_[kNHWC_H] = DIMENSION_1D;
    perm_4d_[kNHWC_W] = DIMENSION_2D;
    perm_4d_[kNHWC_C] = tensor_size_.AlignAxis(perm[kNHWC_H]);
    if (num_axes != static_cast<int>(tensor_size_.NDim)) {
      perm_4d_[kNHWC_N] = DIMENSION_0D;
      perm_4d_[kNHWC_H] = DIMENSION_1D;
      perm_4d_[kNHWC_W] = DIMENSION_2D;
      perm_4d_[kNHWC_C] = DIMENSION_3D;
    }
  } else if (tensor_size_.NDim == DIMENSION_3D) {
    perm_4d_[kNHWC_N] = tensor_size_.AlignAxis(perm[kNHWC_N]);
    perm_4d_[kNHWC_H] = DIMENSION_1D;
    perm_4d_[kNHWC_W] = tensor_size_.AlignAxis(perm[kNHWC_H]);
    perm_4d_[kNHWC_C] = tensor_size_.AlignAxis(perm[kNHWC_W]);
  } else if (tensor_size_.NDim == DIMENSION_4D) {
    perm_4d_[kNHWC_N] = tensor_size_.AlignAxis(perm[kNHWC_N]);
    perm_4d_[kNHWC_H] = tensor_size_.AlignAxis(perm[kNHWC_H]);
    perm_4d_[kNHWC_W] = tensor_size_.AlignAxis(perm[kNHWC_W]);
    perm_4d_[kNHWC_C] = tensor_size_.AlignAxis(perm[kNHWC_C]);
  } else {
    perm_4d_[kNHWC_N] = DIMENSION_0D;
    perm_4d_[kNHWC_H] = DIMENSION_1D;
    perm_4d_[kNHWC_W] = DIMENSION_2D;
    perm_4d_[kNHWC_C] = DIMENSION_3D;
  }
}

int TransposeOpenCLKernel::Prepare() {
  BroadCastPerm();

  std::string kernel_name = "transpose";
  if (tensor_size_.N == 1 && perm_4d_[kNHWC_N] == DIMENSION_0D && perm_4d_[kNHWC_H] == DIMENSION_3D &&
      perm_4d_[kNHWC_W] == DIMENSION_1D && perm_4d_[kNHWC_C] == DIMENSION_2D) {
    type_ = TransposeType::AXIS0312;
    kernel_name += "_0312";
  } else if (tensor_size_.N == 1 && perm_4d_[kNHWC_N] == 0 && perm_4d_[kNHWC_H] == DIMENSION_2D &&
             perm_4d_[kNHWC_W] == DIMENSION_3D && perm_4d_[kNHWC_C] == DIMENSION_1D) {
    type_ = TransposeType::AXIS0231;
    kernel_name += "_0231";
  } else {
    type_ = TransposeType::GENERAL;
    kernel_name += "_general";
  }

  if (in_tensors_[0]->shape().size() == static_cast<int>(DIMENSION_4D) &&
      in_tensors_[0]->shape()[kNHWC_W] * UP_DIV(in_tensors_[0]->shape()[kNHWC_C], C4NUM) >
        static_cast<int>(ocl_runtime_->GetMaxImage2DWidth())) {
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
  (void)SetGlobalLocal();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int TransposeOpenCLKernel::SetConstArgs() {
  size_t n = tensor_size_.N;
  size_t h = tensor_size_.H;
  size_t w = tensor_size_.W;
  size_t c = tensor_size_.C;
  int arg_idx = CLARGSINDEX2;
  cl_int4 shape = {static_cast<int>(n), static_cast<int>(h), static_cast<int>(w), static_cast<int>(c)};
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, shape) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (type_ == TransposeType::GENERAL) {
    int de_perm[C4NUM];  // output to input perm
    for (int i = 0; i < C4NUM; i++) {
      de_perm[perm_4d_[i]] = i;
    }
    cl_int4 de_perm_cl = {de_perm[kNHWC_N], de_perm[kNHWC_H], de_perm[kNHWC_W], de_perm[kNHWC_C]};
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

int TransposeOpenCLKernel::SetGlobalLocal() {
  size_t n = tensor_size_.N;
  size_t h = tensor_size_.H;
  size_t w = tensor_size_.W;
  size_t c = tensor_size_.C;
  size_t c4 = UP_DIV(c, C4NUM);
  local_size_ = {};
  if (type_ == TransposeType::AXIS0312) {  // NHWC -> NCHW
    global_size_ = {UP_DIV(h, C4NUM), w, c4};
  } else if (type_ == TransposeType::AXIS0231) {  // NCHW -> NHWC
    global_size_ = {h, UP_DIV(w, C4NUM), c4};
  } else {  // general
    global_size_ = {n * h, w, c4};
  }
  AlignGlobalLocal(global_size_, local_size_);

  return RET_OK;
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
