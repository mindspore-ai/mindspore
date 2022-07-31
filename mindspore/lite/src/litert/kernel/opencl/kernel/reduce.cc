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
#include <algorithm>
#include "include/errorcode.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/opencl/kernel/reduce.h"
#include "src/litert/kernel/opencl/utils.h"
#include "src/litert/kernel/opencl/cl/reduce.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::PrimitiveType_ReduceFusion;
using mindspore::schema::ReduceMode;
using mindspore::schema::ReduceMode_ReduceMax;
using mindspore::schema::ReduceMode_ReduceMean;
using mindspore::schema::ReduceMode_ReduceMin;
using mindspore::schema::ReduceMode_ReduceProd;
using mindspore::schema::ReduceMode_ReduceSum;
using mindspore::schema::ReduceMode_ReduceSumSquare;

namespace mindspore::kernel {
std::string ReduceOpenCLKernel::GetReduceTypeStr(int type) {
  static const std::map<int, std::string> reduce_type2str{
    {ReduceMode_ReduceMean, "Mean"}, {ReduceMode_ReduceSum, "Sum"},   {ReduceMode_ReduceMin, "Min"},
    {ReduceMode_ReduceMax, "Max"},   {ReduceMode_ReduceProd, "Prod"}, {ReduceMode_ReduceSumSquare, "SumSquare"}};
  auto result_iter = reduce_type2str.find(type);
  if (result_iter != reduce_type2str.end()) {
    return result_iter->second;
  }
  return "";
}

cl_float4 ReduceOpenCLKernel::GenC4Mask() {
  auto reduce_param = reinterpret_cast<ReduceParameter *>(op_parameter_);
  int last_c4 = inShape.C % C4NUM;
  if (last_c4 == 0) last_c4 = C4NUM;
  static const std::map<int, float> reduce_type2init{
    {ReduceMode_ReduceMean, 0.f},     {ReduceMode_ReduceSum, 0.f},  {ReduceMode_ReduceMin, 10000.f},
    {ReduceMode_ReduceMax, -10000.f}, {ReduceMode_ReduceProd, 1.f}, {ReduceMode_ReduceSumSquare, 0.f}};
  float init_float = reduce_type2init.find(reduce_param->mode_)->second;
  cl_float4 mask = {0.f, 0.f, 0.f, 0.f};
  for (int i = 0; i < last_c4; i++) {
    mask.s[C4NUM - i - 1] = init_float;
  }
  return mask;
}

bool ReduceOpenCLKernel::IsHWCReduce() {
  return !reduce_axes_[kNHWC_N] && reduce_axes_[kNHWC_H] && reduce_axes_[kNHWC_W] && reduce_axes_[kNHWC_C];
}

bool ReduceOpenCLKernel::IsHWReduce() {
  return !reduce_axes_[kNHWC_N] && reduce_axes_[kNHWC_H] && reduce_axes_[kNHWC_W] && !reduce_axes_[kNHWC_C];
}

bool ReduceOpenCLKernel::IsWCReduce() {
  return !reduce_axes_[kNHWC_N] && !reduce_axes_[kNHWC_H] && reduce_axes_[kNHWC_W] && reduce_axes_[kNHWC_C];
}

bool ReduceOpenCLKernel::IsHReduce() {
  return !reduce_axes_[kNHWC_N] && reduce_axes_[kNHWC_H] && !reduce_axes_[kNHWC_W] && !reduce_axes_[kNHWC_C];
}

bool ReduceOpenCLKernel::IsWReduce() {
  return !reduce_axes_[kNHWC_N] && !reduce_axes_[kNHWC_H] && reduce_axes_[kNHWC_W] && !reduce_axes_[kNHWC_C];
}

bool ReduceOpenCLKernel::IsCReduce() {
  return !reduce_axes_[kNHWC_N] && !reduce_axes_[kNHWC_H] && !reduce_axes_[kNHWC_W] && reduce_axes_[kNHWC_C];
}

int ReduceOpenCLKernel::SetShapeSizeIs0Axes() {
  // axes is input tensor
  auto *axes_tensor = in_tensors_.at(1);
  auto input_shape_size = in_tensors_.at(0)->shape().size();
  if (input_shape_size == 0) {
    return RET_ERROR;
  }

  CHECK_NULL_RETURN(axes_tensor->data());

  auto reduction_indices = reinterpret_cast<int *>(axes_tensor->data())[0];
  if (reduction_indices == -1) {
    reduce_axes_[kNHWC_H] = true;
    reduce_axes_[kNHWC_W] = true;
    reduce_axes_[kNHWC_C] = true;
  } else if (reduction_indices == kNHWC_H || reduction_indices == kNHWC_W || reduction_indices == kNHWC_C) {
    reduction_indices = reduction_indices + (C4NUM % input_shape_size);
    reduce_axes_[reduction_indices] = true;
  } else {
    MS_LOG(WARNING) << "in Reduce: axes tensor's reduction_indices should be -1, 1, 2, 3";
    return RET_ERROR;
  }
  return RET_OK;
}

int ReduceOpenCLKernel::SetShapeSizeIs1Axes() {
  // axes is input tensor
  // get num_axes
  auto *axes_tensor = in_tensors_.at(1);
  int num_axes = axes_tensor->shape().front();
  // check axes tensor
  if (CheckParamLikeTensor("Reduce", "axes", axes_tensor, kNumberTypeInt32, {num_axes}) != RET_OK) {
    return RET_ERROR;
  }
  // copy axes from tensor to private var
  CHECK_NULL_RETURN(axes_tensor->data());
  for (int i = 0; i < std::min(num_axes, MAX_SHAPE_SIZE); ++i) {
    axes_[i] = reinterpret_cast<int *>(axes_tensor->data())[i];
  }
  if (num_axes > C2NUM || num_axes < C1NUM) {
    MS_LOG(WARNING) << "Unsupported reduce num axes " << num_axes;
    return RET_PARAM_INVALID;
  }

  for (int i = 0; i < num_axes; i++) {
    int axis = axes_[i];
    axis = inShape.AlignAxis(axis);
    reduce_axes_[axis] = true;
  }
  if (num_axes == 1) {
    if (reduce_axes_[kNHWC_H] && inShape.W == 1) {
      reduce_axes_[kNHWC_W] = true;
    } else if (reduce_axes_[kNHWC_W]) {
      if (inShape.H == 1) {
        reduce_axes_[kNHWC_H] = true;
      } else if (inShape.C == 1) {
        reduce_axes_[kNHWC_C] = true;
      }
    } else if (reduce_axes_[kNHWC_C] && inShape.W == 1) {
      reduce_axes_[kNHWC_C] = true;
    }
  }
  return RET_OK;
}

int ReduceOpenCLKernel::SetAxes() {
  auto *axes_tensor = in_tensors_.at(1);

  if (axes_tensor->shape().size() == 0) {
    return SetShapeSizeIs0Axes();
  } else if (axes_tensor->shape().size() == 1) {
    return SetShapeSizeIs1Axes();
  } else {
    MS_LOG(WARNING) << "in Reduce: axes tensor's ndim should be 0 or 1.";
    return RET_ERROR;
  }

  return RET_OK;
}

int ReduceOpenCLKernel::IsReduceAxesSupport() {
  if (!IsHWReduce() && !IsWCReduce() && !IsHReduce() && !IsWReduce() && !IsCReduce() && !IsHWCReduce()) {
    MS_LOG(WARNING) << "Unsupported reduce axes";
    return RET_PARAM_INVALID;
  }
  return RET_OK;
}

int ReduceOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_2 || out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  auto input = in_tensors_.at(0);
  CHECK_NULL_RETURN(input);
  if (input->shape()[0] > DIMENSION_1D) {
    MS_LOG(WARNING) << "reduce op only support n = 1";
    return RET_PARAM_INVALID;
  }
  inShape = GpuTensorInfo(in_tensors_[0]);
  auto reduce_param = reinterpret_cast<ReduceParameter *>(op_parameter_);
  CHECK_NULL_RETURN(reduce_param);
  if (GetReduceTypeStr(reduce_param->mode_).empty()) {
    MS_LOG(WARNING) << "not supported reduce type:" << reduce_param->mode_;
    return RET_PARAM_INVALID;
  }
  auto ret = SetAxes();
  if (ret != RET_OK) {
    return ret;
  }

  if (IsReduceAxesSupport() != RET_OK) {
    return RET_PARAM_INVALID;
  }
  if (IsWCReduce() && !reduce_param->keep_dims_) {
    MS_LOG(WARNING) << "reduce axis (2,3) should keep dims";
    return RET_PARAM_INVALID;
  }
  return RET_OK;
}

int ReduceOpenCLKernel::Prepare() {
  auto reduce_param = reinterpret_cast<ReduceParameter *>(op_parameter_);
  if (reduce_param == nullptr) {
    return RET_NULL_PTR;
  }

  std::string kernel_name;
  use_local_ = false;
  kernel_name = "Global";
  if (IsWCReduce() && (inShape.W >= LOCAL_CACHE_THREAD || inShape.C >= LOCAL_CACHE_THREAD)) {
    use_local_ = true;
    kernel_name = "Local";
  }
  if (IsHWReduce() && (inShape.W >= LOCAL_CACHE_THREAD || inShape.H >= LOCAL_CACHE_THREAD)) {
    use_local_ = true;
    kernel_name = "Local";
  }

  if (IsHWCReduce()) {
    kernel_name += "HWC";
  } else if (IsWCReduce()) {
    kernel_name += "WC";
  } else if (IsHWReduce()) {
    kernel_name += "HW";
  } else if (IsHReduce()) {
    kernel_name += "H";
  } else if (IsWReduce()) {
    kernel_name += "W";
  } else if (IsCReduce()) {
    kernel_name += "C";
  }
  kernel_name += GetReduceTypeStr(reduce_param->mode_);
  std::string source = reduce_source;
  const std::string program_name = "Reduce";
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

int ReduceOpenCLKernel::SetConstArgs() {
  int h = inShape.H;
  int w = inShape.W;
  int c = inShape.C;
  int c4 = UP_DIV(c, C4NUM);
  cl_int4 size = {h, w, c4, c};
  int arg_idx = 2;
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, size) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (IsWCReduce() || IsCReduce()) {
    if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, GenC4Mask()) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int ReduceOpenCLKernel::SetGlobalLocal() {
  int h = inShape.H;
  int w = inShape.W;
  int c4 = inShape.Slice;
  local_size_ = {};
  if (use_local_) {
    local_size_ = {1, LOCAL_CACHE_THREAD, LOCAL_CACHE_THREAD};
  }
  if (IsHWCReduce()) {
    global_size_ = {1, 1, 1};
  } else if (IsHWReduce()) {
    global_size_ = {static_cast<size_t>(c4), 1, 1};
  } else if (IsWCReduce()) {
    global_size_ = {static_cast<size_t>(h), 1, 1};
  } else if (IsHReduce()) {
    global_size_ = {static_cast<size_t>(w), static_cast<size_t>(c4)};
  } else if (IsWReduce()) {
    global_size_ = {static_cast<size_t>(h), static_cast<size_t>(c4)};
  } else if (IsCReduce() && !use_local_) {
    global_size_ = {static_cast<size_t>(h), static_cast<size_t>(w)};
  } else {
    global_size_ = {1, 1, 1};
  }

  AlignGlobalLocal(global_size_, local_size_);

  return RET_OK;
}

int ReduceOpenCLKernel::Tune() {
  if (use_local_) {
    return RET_OK;
  }
  return OpenCLKernel::Tune();
}

int ReduceOpenCLKernel::Run() {
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

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_ReduceFusion, OpenCLKernelCreator<ReduceOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_ReduceFusion, OpenCLKernelCreator<ReduceOpenCLKernel>)
}  // namespace mindspore::kernel
