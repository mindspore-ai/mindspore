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

#include "src/runtime/kernel/arm/base/argminmax_base.h"
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
  auto input = in_tensors_.at(0);
  auto shape = input->shape();

  auto input_data = input->data_c();
  auto output_data = out_tensors_.at(0)->data_c();
  void *output_value = nullptr;
  if (out_tensors_.size() == 2) {
    output_value = out_tensors_.at(1)->data_c();
  }

  MS_ASSERT(context_->allocator != nullptr);
  if (arg_param_->topk_ > 1 || arg_param_->keep_dims_) {
    arg_param_->arg_elements_ =
      reinterpret_cast<ArgElement *>(context_->allocator->Malloc(sizeof(ArgElement) * shape[arg_param_->axis_]));
    if (arg_param_->arg_elements_ == nullptr) {
      MS_LOG(ERROR) << "malloc memory fail!";
      return RET_ERROR;
    }
  }
  if (input->data_type() == kNumberTypeFloat32) {
    ArgMinMaxFp32(reinterpret_cast<float *>(input_data), reinterpret_cast<void *>(output_data),
                  reinterpret_cast<float *>(output_value), shape.data(), arg_param_);
#ifdef ENABLE_ARM64
  } else if (input->data_type() == kNumberTypeFloat16) {
    ArgMinMaxFp16(reinterpret_cast<float16_t *>(input_data), reinterpret_cast<void *>(output_data),
                  reinterpret_cast<float16_t *>(output_value), shape.data(), arg_param_);

#endif
  } else {
    MS_LOG(ERROR) << "unsupported data type!";
  }

  context_->allocator->Free(arg_param_->arg_elements_);
  arg_param_->arg_elements_ = nullptr;
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ArgMaxFusion, LiteKernelCreator<ArgMinMaxCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ArgMinFusion, LiteKernelCreator<ArgMinMaxCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_ArgMaxFusion, LiteKernelCreator<ArgMinMaxCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_ArgMinFusion, LiteKernelCreator<ArgMinMaxCPUKernel>)
}  // namespace mindspore::kernel
