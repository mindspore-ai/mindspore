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

#include "src/runtime/kernel/arm/fp32/argminmax_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ArgMaxFusion;
using mindspore::schema::PrimitiveType_ArgMinFusion;

namespace mindspore::kernel {
int ArgMinMaxCPUKernel::Init() {
  arg_param_->data_type_ = kNumberTypeFloat32;
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ArgMinMaxCPUKernel::ReSize() {
  auto in_shape = in_tensors_.at(0)->shape();
  auto dims_size = in_shape.size();
  int axis = arg_param_->axis_ < 0 ? arg_param_->axis_ + dims_size : arg_param_->axis_;
  arg_param_->axis_ = axis;
  arg_param_->dims_size_ = dims_size;
  if (arg_param_->topk_ <= 0) {
    MS_LOG(ERROR) << "Invalid topk " << arg_param_->topk_;
    return RET_ERROR;
  }
  arg_param_->topk_ = MSMIN(arg_param_->topk_, in_shape.at(axis));
  ComputeStrides(in_shape.data(), arg_param_->in_strides_, in_shape.size());
  auto out_shape = out_tensors_.at(0)->shape();
  ComputeStrides(out_shape.data(), arg_param_->out_strides_, out_shape.size());
  return RET_OK;
}

int ArgMinMaxCPUKernel::Run() {
  float *input_data = reinterpret_cast<float *>(in_tensors_.at(0)->data_c());
  int *output_index = nullptr;
  float *output_value = nullptr;
  if (out_tensors_.size() == 2) {
    output_index = reinterpret_cast<int *>(out_tensors_.at(0)->data_c());
    output_value = reinterpret_cast<float *>(out_tensors_.at(1)->data_c());
  } else if (arg_param_->out_value_) {
    output_value = reinterpret_cast<float *>(out_tensors_.at(0)->data_c());
  } else {
    output_index = reinterpret_cast<int *>(out_tensors_.at(0)->data_c());
  }

  auto shape = in_tensors_.at(0)->shape();

  MS_ASSERT(context_->allocator != nullptr);
  if (arg_param_->topk_ > 1 || arg_param_->keep_dims_) {
    arg_param_->arg_elements_ =
      reinterpret_cast<ArgElement *>(context_->allocator->Malloc(sizeof(ArgElement) * shape[arg_param_->axis_]));
    if (arg_param_->arg_elements_ == nullptr) {
      MS_LOG(ERROR) << "malloc memroy fail!";
      return RET_ERROR;
    }
  }
  ArgMinMaxFp32(input_data, output_index, output_value, reinterpret_cast<const int *>(shape.data()), arg_param_);
  context_->allocator->Free(arg_param_->arg_elements_);
  arg_param_->arg_elements_ = nullptr;
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ArgMaxFusion, LiteKernelCreator<ArgMinMaxCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ArgMinFusion, LiteKernelCreator<ArgMinMaxCPUKernel>)
}  // namespace mindspore::kernel
