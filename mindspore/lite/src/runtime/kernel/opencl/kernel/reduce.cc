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
#include <algorithm>
#include <string>
#include <map>
#include "include/errorcode.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/kernel/reduce.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/kernel/opencl/cl/reduce.cl.inc"

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

bool IsHWReduce(bool *reduce_axes_) {
  return !reduce_axes_[0] && reduce_axes_[1] && reduce_axes_[2] && !reduce_axes_[3];
}

bool IsWCReduce(bool *reduce_axes_) {
  return !reduce_axes_[0] && !reduce_axes_[1] && reduce_axes_[2] && reduce_axes_[3];
}

bool IsCReduce(bool *reduce_axes_) {
  return !reduce_axes_[0] && !reduce_axes_[1] && !reduce_axes_[2] && reduce_axes_[3];
}

int ReduceOpenCLKernel::SetAxes() {
  // axes is input tensor
  // get num_axes
  int num_axes = 0;
  auto *axes_tensor = in_tensors_.at(1);
  if (axes_tensor->shape().size() != 1) {
    MS_LOG(ERROR) << "in Reduce: axes tensor's ndim should be 1.";
    return RET_ERROR;
  } else {
    num_axes = axes_tensor->shape().front();
  }
  // check axes tensor
  if (CheckParamLikeTensor("Reduce", "axes", axes_tensor, kNumberTypeInt32, {num_axes}) != RET_OK) {
    return RET_ERROR;
  }
  // copy axes from tensor to private var
  for (int i = 0; i < std::min(num_axes, MAX_SHAPE_SIZE); ++i) {
    axes_[i] = reinterpret_cast<int *>(axes_tensor->data_c())[i];
  }
  if (num_axes > 2 || num_axes < 1) {
    MS_LOG(ERROR) << "Unsupported reduce num axes " << num_axes;
    return RET_PARAM_INVALID;
  }

  for (int i = 0; i < num_axes; i++) {
    int axis = axes_[i];
    axis = inShape.AlignAxis(axis);
    reduce_axes_[axis] = true;
  }
  if (num_axes == 1) {
    if (reduce_axes_[1] && inShape.W == 1) {
      reduce_axes_[2] = true;
    } else if (reduce_axes_[2]) {
      if (inShape.H == 1) {
        reduce_axes_[1] = true;
      } else if (inShape.C == 1) {
        reduce_axes_[3] = true;
      }
    } else if (reduce_axes_[3] && inShape.W == 1) {
      reduce_axes_[3] = true;
    }
  }
  return RET_OK;
}

int ReduceOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() != 2 || out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  if (in_tensors_[0]->shape()[0] > 1) {
    MS_LOG(ERROR) << "reduce op only support n = 1";
    return RET_PARAM_INVALID;
  }
  inShape = GpuTensorInfo(in_tensors_[0]);
  auto reduce_param = reinterpret_cast<ReduceParameter *>(op_parameter_);
  if (GetReduceTypeStr(reduce_param->mode_).empty()) {
    MS_LOG(ERROR) << "not supported reduce type:" << reduce_param->mode_;
    return RET_PARAM_INVALID;
  }
  auto ret = SetAxes();
  if (ret != RET_OK) return ret;
  hw_reduce_ = IsHWReduce(reduce_axes_);
  wc_reduce_ = IsWCReduce(reduce_axes_);
  c_reduce_ = IsCReduce(reduce_axes_);
  if (!hw_reduce_ && !wc_reduce_ && !c_reduce_) {
    MS_LOG(ERROR) << "Unsupported reduce axes";
    return RET_PARAM_INVALID;
  }
  if ((c_reduce_ || wc_reduce_) && reduce_param->keep_dims_ == false) {
    MS_LOG(ERROR) << "reduce axis (2,3) should keep dims";
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
  if (wc_reduce_ && (inShape.W >= LOCAL_CACHE_THREAD || inShape.C >= LOCAL_CACHE_THREAD)) {
    use_local_ = true;
    kernel_name = "Local";
  }
  if (hw_reduce_ && (inShape.W >= LOCAL_CACHE_THREAD || inShape.H >= LOCAL_CACHE_THREAD)) {
    use_local_ = true;
    kernel_name = "Local";
  }
  if (wc_reduce_) {
    kernel_name += "WC";
  } else if (hw_reduce_) {
    kernel_name += "HW";
  } else if (c_reduce_) {
    kernel_name += "C";
  }
  kernel_name += GetReduceTypeStr(reduce_param->mode_);
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else
  std::string source = reduce_source;
  std::string program_name = "Reduce";
  ocl_runtime_->LoadSource(program_name, source);
  auto ret = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);
  if (ret != RET_OK) {
    return ret;
  }
#endif
  SetConstArgs();
  SetGlobalLocal();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return mindspore::lite::RET_OK;
}
void ReduceOpenCLKernel::SetConstArgs() {
  int h = inShape.H;
  int w = inShape.W;
  int c = inShape.C;
  int c4 = UP_DIV(c, C4NUM);
  cl_int4 size = {h, w, c4, c};
  int arg_idx = 2;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, size);
  if (wc_reduce_ || c_reduce_) {
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, GenC4Mask());
  }
}
void ReduceOpenCLKernel::SetGlobalLocal() {
  int h = inShape.H;
  int w = inShape.W;
  int c4 = inShape.Slice;
  local_size_ = {};
  if (use_local_) {
    local_size_ = {1, LOCAL_CACHE_THREAD, LOCAL_CACHE_THREAD};
  }
  if (hw_reduce_) {
    global_size_ = {static_cast<size_t>(c4), 1, 1};
  } else if (wc_reduce_) {
    global_size_ = {static_cast<size_t>(h), 1, 1};
  } else if (c_reduce_ && !use_local_) {
    global_size_ = {static_cast<size_t>(h), static_cast<size_t>(w)};
  }
  AlignGlobalLocal(global_size_, local_size_);
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
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->data_c());
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return mindspore::lite::RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_ReduceFusion, OpenCLKernelCreator<ReduceOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_ReduceFusion, OpenCLKernelCreator<ReduceOpenCLKernel>)
}  // namespace mindspore::kernel
