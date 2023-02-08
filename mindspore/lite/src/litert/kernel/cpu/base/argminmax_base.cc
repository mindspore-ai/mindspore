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

#include "src/litert/kernel/cpu/base/argminmax_base.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ArgMaxFusion;
using mindspore::schema::PrimitiveType_ArgMinFusion;

namespace mindspore::kernel {
int ArgMinMaxCPUKernel::Prepare() {
  CHECK_NOT_EQUAL_RETURN(in_tensors_.size(), C1NUM);
  CHECK_LARGE_RETURN(out_tensors_.size(), C2NUM);
  CHECK_NULL_RETURN(arg_param_);
  arg_param_->data_type_ = kNumberTypeFloat32;
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ArgMinMaxCPUKernel::ReSize() {
  CHECK_NULL_RETURN(in_tensors_.at(0));
  auto in_shape = in_tensors_.at(0)->shape();
  auto dims_size = in_shape.size();
  MS_CHECK_TRUE_MSG(dims_size >= 0, RET_ERROR, "The input shape is invalid.");
  arg_param_->dims_size_ = static_cast<int>(dims_size);
  int axis = arg_param_->axis_ < 0 ? (arg_param_->axis_ + arg_param_->dims_size_) : arg_param_->axis_;
  arg_param_->axis_ = axis;
  MS_CHECK_TRUE_MSG(axis >= 0 && axis < static_cast<int>(in_shape.size()), RET_ERROR, "The axis is invalid.");
  if (arg_param_->topk_ <= 0 || arg_param_->topk_ > in_shape.at(axis)) {
    MS_LOG(ERROR) << "Invalid topk " << arg_param_->topk_;
    return RET_ERROR;
  }
  CHECK_NULL_RETURN(in_shape.data());
  ComputeStrides(in_shape.data(), arg_param_->in_strides_, in_shape.size());
  CHECK_NULL_RETURN(out_tensors_.at(0));
  auto out_shape = out_tensors_.at(0)->shape();
  MS_CHECK_TRUE_MSG(out_shape.size() >= 0, RET_ERROR, "The out shape is invalid.");
  CHECK_NULL_RETURN(out_shape.data());
  MS_CHECK_TRUE_MSG(static_cast<int>(out_shape.size()) <= COMM_SHAPE_SIZE, RET_ERROR, "The out_shape size invalid.");
  ComputeStrides(out_shape.data(), arg_param_->out_strides_, out_shape.size());
  return RET_OK;
}

int ArgMinMaxCPUKernel::Run() {
  auto input = in_tensors_.at(0);
  CHECK_NULL_RETURN(input);
  auto shape = input->shape();
  MS_CHECK_TRUE_MSG(arg_param_->axis_ >= 0 && arg_param_->axis_ < static_cast<int>(shape.size()), RET_ERROR,
                    "The axis is invalid.");

  auto input_data = input->data();
  auto output_data = out_tensors_.at(0)->data();
  if (input_data == nullptr || output_data == nullptr) {
    return RET_NULL_PTR;
  }
  CHECK_NULL_RETURN(shape.data());
  void *output_value = nullptr;
  if (out_tensors_.size() == C2NUM) {
    output_value = out_tensors_.at(1)->data();
    if (output_value == nullptr) {
      return RET_NULL_PTR;
    }
  }

  CHECK_NULL_RETURN(ms_context_->allocator);
  if (arg_param_->topk_ > C1NUM || arg_param_->keep_dims_) {
    MS_CHECK_FALSE(INT_MUL_OVERFLOW(static_cast<int>(sizeof(ArgElement)), shape[arg_param_->axis_]), RET_ERROR);
    arg_param_->arg_elements_ =
      reinterpret_cast<ArgElement *>(ms_context_->allocator->Malloc(sizeof(ArgElement) * shape[arg_param_->axis_]));
    if (arg_param_->arg_elements_ == nullptr) {
      MS_LOG(ERROR) << "malloc memory fail!";
      return RET_ERROR;
    }
  }
  if (input->data_type() == kNumberTypeFloat32) {
    ArgMinMaxFp32(reinterpret_cast<float *>(input_data), reinterpret_cast<void *>(output_data),
                  reinterpret_cast<float *>(output_value), shape.data(), arg_param_);
#ifdef ENABLE_FP16
  } else if (input->data_type() == kNumberTypeFloat16) {
    ArgMinMaxFp16(reinterpret_cast<float16_t *>(input_data), reinterpret_cast<void *>(output_data),
                  reinterpret_cast<float16_t *>(output_value), shape.data(), arg_param_);
#endif
  } else {
    MS_LOG(ERROR) << "unsupported data type!";
    ms_context_->allocator->Free(arg_param_->arg_elements_);
    arg_param_->arg_elements_ = nullptr;
    return RET_ERROR;
  }

  ms_context_->allocator->Free(arg_param_->arg_elements_);
  arg_param_->arg_elements_ = nullptr;
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ArgMaxFusion, LiteKernelCreator<ArgMinMaxCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ArgMinFusion, LiteKernelCreator<ArgMinMaxCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_ArgMaxFusion, LiteKernelCreator<ArgMinMaxCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_ArgMinFusion, LiteKernelCreator<ArgMinMaxCPUKernel>)
}  // namespace mindspore::kernel
