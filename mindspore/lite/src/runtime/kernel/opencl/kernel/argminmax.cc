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
#include <cstring>
#include <string>
#include <algorithm>
#include <set>
#include <utility>
#include <functional>
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/kernel/opencl/kernel/argminmax.h"
#include "src/runtime/kernel/opencl/cl/argminmax.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ArgMaxFusion;
using mindspore::schema::PrimitiveType_ArgMinFusion;

namespace mindspore::kernel {

int ArgMinMaxOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() != 1 || out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  if ((in_tensors_[0]->data_type() != kNumberTypeFloat32 && in_tensors_[0]->data_type() != kNumberTypeFloat16) ||
      (out_tensors_[0]->data_type() != kNumberTypeFloat32 && out_tensors_[0]->data_type() != kNumberTypeFloat16)) {
    MS_LOG(ERROR) << "Unsupported data type " << in_tensors_[0]->data_type();
    return RET_ERROR;
  }
  if (in_tensors_[0]->shape().size() > 4 && in_tensors_[0]->shape().size() == 0) {
    MS_LOG(ERROR) << "input shape size must be (1-4), actual: " << in_tensors_[0]->shape().size() << ", "
                  << out_tensors_[0]->shape().size();
    return RET_ERROR;
  }
  auto *param = reinterpret_cast<ArgMinMaxParameter *>(this->op_parameter_);
  auto dims_size = in_tensors_[0]->shape().size();
  auto axis = (param->axis_ + dims_size) % dims_size;
  if (axis < 0 || axis >= dims_size) {
    MS_LOG(ERROR) << "Invalid axis " << axis;
    return RET_ERROR;
  }
  return RET_OK;
}

void ArgMinMaxOpenCLKernel::SetConstArgs() {
  auto param = reinterpret_cast<ArgMinMaxParameter *>(op_parameter_);
  cl_int4 in_shape{static_cast<int>(im_in_.N), static_cast<int>(im_in_.H), static_cast<int>(im_in_.W),
                   static_cast<int>(im_in_.C)};
  cl_int4 flags = {param->out_value_, param->get_max_, param->axis_, param->topk_};
  int arg_cnt = 2;
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, buff_, lite::opencl::MemType::BUF);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, ids_, lite::opencl::MemType::BUF);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, in_shape);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, src_size_);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, cus_size_);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, strides_);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, flags);
}

void ArgMinMaxOpenCLKernel::SetGlobalLocal() {
  auto param = reinterpret_cast<ArgMinMaxParameter *>(op_parameter_);
  im_in_ = GpuTensorInfo(in_tensors_[0]);
  im_out_ = GpuTensorInfo(out_tensors_[0]);
  std::vector<size_t> in_shape = {im_in_.N, im_in_.H, im_in_.W, im_in_.C};
  auto in_shape_align = in_shape;
  in_shape_align[3] = UP_ROUND(in_shape[3], C4NUM);
  std::vector<size_t> out_shape = {im_out_.N, im_out_.H, im_out_.W, im_out_.C};
  auto out_shape_align = out_shape;
  out_shape_align[3] = UP_ROUND(out_shape[3], C4NUM);
  int reduce_len = GetUpPow2(in_shape.at(param->axis_));
  int dtype_size = in_tensors_[0]->data_type() == kNumberTypeFloat16 ? sizeof(int16_t) : sizeof(float);
  int in_pitch = im_in_.RowPitch() / dtype_size;
  int out_pitch = im_out_.RowPitch() / dtype_size;
  cus_size_ = {reduce_len, param->keep_dims_, 1, 1};
  cus_size_.s[2] = in_pitch - im_in_.width * C4NUM;
  cus_size_.s[3] = out_pitch - im_out_.width * C4NUM;
  src_size_ = {std::accumulate(in_shape.begin() + param->axis_ + 1, in_shape.end(), 1, std::multiplies<int>()),
               std::accumulate(in_shape.begin(), in_shape.begin() + param->axis_, 1, std::multiplies<int>()),
               std::accumulate(in_shape.begin() + param->axis_, in_shape.end(), 1, std::multiplies<int>()),
               static_cast<int>(in_shape.at(param->axis_))};
  int out_axis = (param->axis_ == 3 && param->topk_ == 1 && !param->keep_dims_) ? 4 : param->axis_;
  strides_ = {
    std::accumulate(in_shape_align.begin() + param->axis_ + 1, in_shape_align.end(), 1, std::multiplies<int>()),
    std::accumulate(in_shape_align.begin() + param->axis_, in_shape_align.end(), 1, std::multiplies<int>()),
    std::accumulate(out_shape_align.begin() + std::min(out_axis + 1, 4), out_shape_align.end(), 1,
                    std::multiplies<int>()),
    std::accumulate(out_shape_align.begin() + out_axis, out_shape_align.end(), 1, std::multiplies<int>()),
  };
  switch (param->axis_) {
    case 0:
      strides_.s[0] = UP_ROUND(strides_.s[0] / im_in_.H, in_pitch) * im_in_.H;
      strides_.s[1] = strides_.s[0] * im_in_.N;
      strides_.s[2] = UP_ROUND(strides_.s[2] / im_in_.H, out_pitch) * im_in_.H;
      strides_.s[3] = strides_.s[2] * param->topk_;
      break;
    case 1:
      strides_.s[0] = UP_ROUND(strides_.s[0], in_pitch);
      strides_.s[1] = UP_ROUND(strides_.s[1] / im_in_.H, in_pitch) * im_in_.H;
      // org dim(4,3) org axis(1,0)
      strides_.s[2] = UP_ROUND(strides_.s[2], out_pitch);
      strides_.s[3] = UP_ROUND(strides_.s[3] / param->topk_, out_pitch) * param->topk_;
      break;
    case 2:
      strides_.s[1] = UP_ROUND(strides_.s[1], in_pitch);
      // org dim(4,3,2) org axis(2,1,0)
      strides_.s[3] = param->keep_dims_ ? UP_ROUND(strides_.s[3], out_pitch) : strides_.s[2];
      break;
    default:  // 3
      // org dim(4,3,2,1) org axis(3,2,1,0)
      break;
  }
  local_size_ = {1, 1, 1};
  global_size_ = {static_cast<size_t>(strides_.s[0]), static_cast<size_t>(src_size_.s[1]), 1};
  OpenCLKernel::AlignGlobalLocal(global_size_, local_size_);
}

int ArgMinMaxOpenCLKernel::InitWeights() {
  auto allocator = ocl_runtime_->GetAllocator();
  int dtype_size = ocl_runtime_->GetFp16Enable() ? sizeof(int16_t) : sizeof(float);
  buff_ = allocator->Malloc(in_tensors_[0]->ElementsNum() * dtype_size);
  ids_ = allocator->Malloc(in_tensors_[0]->ElementsNum() * sizeof(int32_t));
  return RET_OK;
}

int ArgMinMaxOpenCLKernel::Prepare() {
  std::string kernel_name = "argminmax";

#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else

  std::string source = argminmax_source;
  std::string program_name = "argminmax";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);
#endif

  auto *param = reinterpret_cast<ArgMinMaxParameter *>(this->op_parameter_);
  param->dims_size_ = in_tensors_[0]->shape().size();
  param->axis_ = (param->axis_ + param->dims_size_) % param->dims_size_;
  param->axis_ = GetBroadcastGpuAxis(param->dims_size_, param->axis_);
  param->get_max_ = (Type() == PrimitiveType_ArgMaxFusion);
  param->keep_dims_ =
    param->keep_dims_ || param->topk_ > 1 || in_tensors_[0]->shape().size() == out_tensors_[0]->shape().size();

  InitWeights();
  SetGlobalLocal();
  SetConstArgs();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int ArgMinMaxOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  ocl_runtime_->SetKernelArg(kernel_, 0, in_tensors_[0]->data_c(), lite::opencl::MemType::BUF);
  ocl_runtime_->SetKernelArg(kernel_, 1, out_tensors_[0]->data_c(), lite::opencl::MemType::BUF);
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_ArgMinFusion, OpenCLKernelCreator<ArgMinMaxOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_ArgMinFusion, OpenCLKernelCreator<ArgMinMaxOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_ArgMaxFusion, OpenCLKernelCreator<ArgMinMaxOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_ArgMaxFusion, OpenCLKernelCreator<ArgMinMaxOpenCLKernel>);
}  // namespace mindspore::kernel
