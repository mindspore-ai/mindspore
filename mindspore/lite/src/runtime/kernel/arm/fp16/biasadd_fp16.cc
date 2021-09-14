/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <vector>
#include "include/errorcode.h"
#include "schema/model_generated.h"
#include "src/runtime/kernel/arm/fp16/biasadd_fp16.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BiasAdd;

namespace mindspore::kernel {
int BiasAddCPUFp16Kernel::ReSize() {
  auto dims = in_tensors_.at(0)->shape();
  bias_param_->ndim_ = dims.size();
  if (bias_param_->ndim_ < 1 || bias_param_->ndim_ > 5) {
    MS_LOG(ERROR) << "input shape is invalid";
    return RET_ERROR;
  }
  for (size_t i = 0; i < bias_param_->ndim_; i++) {
    bias_param_->in_shape0_[i] = dims[i];
    bias_param_->in_shape1_[i] = 1;
    bias_param_->out_shape_[i] = dims[i];
  }
  bias_param_->in_shape1_[bias_param_->ndim_ - 1] = dims[bias_param_->ndim_ - 1];
  return RET_OK;
}

int BiasAddCPUFp16Kernel::Run() {
  if (bias_data_ == nullptr) {
    auto ret = GetBiasData();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "GetBiasData is error in run!";
      return ret;
    }
  }
  if (op_parameter_->is_train_session_) {
    if ((IsTrainable() && (IsTrain() || IsRepack())) || (bias_data_type_ == kNumberTypeFloat16)) {
      PackWeight();
      is_repack_ = false;
    }
  }
  auto in = reinterpret_cast<float16_t *>(in_tensors_.at(0)->data());
  auto out = reinterpret_cast<float16_t *>(out_tensors_.at(0)->data());
  CHECK_NULL_RETURN(in);
  CHECK_NULL_RETURN(out);
  size_t data_size = in_tensors_.at(0)->ElementsNum();
  CHECK_NULL_RETURN(ms_context_->allocator);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(data_size, sizeof(float16_t), RET_ERROR);
  auto tile_in = reinterpret_cast<float16_t *>(ms_context_->allocator->Malloc(data_size * sizeof(float16_t)));
  auto tile_bias = reinterpret_cast<float16_t *>(ms_context_->allocator->Malloc(data_size * sizeof(float16_t)));
  if (tile_in == nullptr || tile_bias == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    ms_context_->allocator->Free(tile_in);
    ms_context_->allocator->Free(tile_bias);
    return RET_NULL_PTR;
  }
  auto ret = BroadcastAddFp16(in, bias_data_, tile_in, tile_bias, out, data_size, bias_param_);
  ms_context_->allocator->Free(tile_in);
  ms_context_->allocator->Free(tile_bias);
  return ret;
}

BiasAddCPUFp16Kernel::~BiasAddCPUFp16Kernel() {
  if ((bias_data_type_ == kNumberTypeFloat || bias_data_type_ == kNumberTypeFloat32) && bias_data_ != nullptr) {
    free(bias_data_);
    bias_data_ = nullptr;
  }
}

int BiasAddCPUFp16Kernel::GetBiasData() {
  bias_data_type_ = bias_tensor_->data_type();
  if (bias_data_type_ == kNumberTypeFloat || bias_data_type_ == kNumberTypeFloat32) {
    if (bias_data_ == nullptr) {
      MS_CHECK_INT_MUL_NOT_OVERFLOW(bias_tensor_->ElementsNum(), sizeof(float16_t), RET_ERROR);
      bias_data_ = reinterpret_cast<float16_t *>(malloc(bias_tensor_->ElementsNum() * sizeof(float16_t)));
      if (bias_data_ == nullptr) {
        MS_LOG(ERROR) << "bias_data_ is nullptr";
        return RET_NULL_PTR;
      }
    }
    auto bias = reinterpret_cast<float *>(bias_tensor_->data());
    if (bias == nullptr) {
      MS_LOG(ERROR) << "bias is nullptr!";
      return RET_NULL_PTR;
    }
    for (int i = 0; i < bias_tensor_->ElementsNum(); ++i) {
      bias_data_[i] = static_cast<float16_t>(bias[i]);
    }
  } else {
    bias_data_ = reinterpret_cast<float16_t *>(bias_tensor_->data());
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "bias_data_ is nullptr";
      return RET_NULL_PTR;
    }
  }
  return RET_OK;
}

int BiasAddCPUFp16Kernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 2);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  bias_tensor_ = in_tensors_.at(1);
  CHECK_NULL_RETURN(bias_tensor_);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

void BiasAddCPUFp16Kernel::PackWeight() {
  if (bias_data_type_ == kNumberTypeFloat || bias_data_type_ == kNumberTypeFloat32) {
    auto bias = reinterpret_cast<float *>(bias_tensor_->data());
    for (int i = 0; i < bias_tensor_->ElementsNum(); ++i) {
      bias_data_[i] = static_cast<float16_t>(bias[i]);
    }
  } else {
    bias_data_ = reinterpret_cast<float16_t *>(bias_tensor_->data());
  }
}

int BiasAddCPUFp16Kernel::Eval() {
  InnerKernel::Eval();
  if (IsTrainable()) {
    is_repack_ = true;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_BiasAdd, LiteKernelCreator<BiasAddCPUFp16Kernel>)
}  // namespace mindspore::kernel
