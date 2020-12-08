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
#include "src/runtime/kernel/opencl/kernel/reduce.h"
#include "src/runtime/kernel/opencl/cl/reduce.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::PrimitiveType_Reduce;
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
  int last_c4 = in_tensors_[0]->shape()[3] % C4NUM;
  last_c4 = (C4NUM - last_c4) % 4;
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

int ReduceOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() != 1 || out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  if (in_tensors_[0]->shape()[0] > 1) {
    MS_LOG(ERROR) << "reduce op only support n = 1";
    return RET_PARAM_INVALID;
  }
  auto reduce_param = reinterpret_cast<ReduceParameter *>(op_parameter_);
  if (GetReduceTypeStr(reduce_param->mode_).empty()) {
    MS_LOG(ERROR) << "not supported reduce type:" << reduce_param->mode_;
    return RET_PARAM_INVALID;
  }
  if (reduce_param->num_axes_ == 1 && reduce_param->axes_[0] == 3 && in_tensors_[0]->shape()[2] == 1) {
    reduce_param->num_axes_ = 2;
    reduce_param->axes_[1] = 2;
  }
  if (reduce_param->num_axes_ != 2) {
    MS_LOG(ERROR) << "reduce op only support axes=2";
    return RET_PARAM_INVALID;
  }
  bool hw_reduce = (reduce_param->axes_[0] == 1 && reduce_param->axes_[1] == 2) ||
                   (reduce_param->axes_[0] == 2 && reduce_param->axes_[1] == 1);
  wc_reduce_ = (reduce_param->axes_[0] == 2 && reduce_param->axes_[1] == 3) ||
               (reduce_param->axes_[0] == 3 && reduce_param->axes_[1] == 2);
  if (!hw_reduce && !wc_reduce_) {
    MS_LOG(ERROR) << "reduce op only support axis (1,2) or (2,3)";
    return RET_PARAM_INVALID;
  }
  if (wc_reduce_ && reduce_param->keep_dims_ == false) {
    MS_LOG(ERROR) << "reduce axis (2,3) should keep dims";
    return RET_PARAM_INVALID;
  }
  return RET_OK;
}

int ReduceOpenCLKernel::Prepare() {
  outShape = GpuTensorInfo(out_tensors_[0]);
  auto reduce_param = reinterpret_cast<ReduceParameter *>(op_parameter_);
  if (reduce_param == nullptr) {
    return RET_NULL_PTR;
  }

  std::string kernel_name;
  if (in_tensors_[0]->shape()[reduce_param->axes_[0]] >= LOCAL_CACHE_THREAD ||
      in_tensors_[0]->shape()[reduce_param->axes_[1]] >= LOCAL_CACHE_THREAD) {
    use_local_ = true;
    kernel_name += "Local";
  } else {
    use_local_ = false;
    kernel_name += "Global";
  }
  if (wc_reduce_) {
    kernel_name += "WC";
  } else {
    kernel_name += "HW";
  }
  kernel_name += GetReduceTypeStr(reduce_param->mode_);
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else
  std::string source = reduce_source;
  std::string program_name = "Reduce";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);
#endif
  SetConstArgs();
  SetGlobalLocal();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return mindspore::lite::RET_OK;
}
void ReduceOpenCLKernel::SetConstArgs() {
  std::vector<int> shapex = in_tensors_[0]->shape();
  int h = shapex[1];
  int w = shapex[2];
  int c = shapex[3];
  int c4 = UP_DIV(c, C4NUM);
  cl_int4 size = {h, w, c4, c};
  int arg_idx = 2;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, size);
  if (wc_reduce_) {
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, GenC4Mask());
  }
}
void ReduceOpenCLKernel::SetGlobalLocal() {
  std::vector<int> shapex = in_tensors_[0]->shape();
  int h = shapex[1];
  int c = shapex[3];
  int c4 = UP_DIV(c, C4NUM);
  local_size_ = {};
  if (use_local_) {
    local_size_ = {1, LOCAL_CACHE_THREAD, LOCAL_CACHE_THREAD};
  }
  global_size_ = {static_cast<size_t>(c4), 1, 1};
  if (wc_reduce_) {
    global_size_ = {static_cast<size_t>(h), 1, 1};
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

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Reduce, OpenCLKernelCreator<ReduceOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Reduce, OpenCLKernelCreator<ReduceOpenCLKernel>)
}  // namespace mindspore::kernel
