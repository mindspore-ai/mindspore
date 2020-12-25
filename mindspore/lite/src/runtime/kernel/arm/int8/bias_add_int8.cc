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

#include "src/runtime/kernel/arm/int8/bias_add_int8.h"
#include "nnacl/fp32/arithmetic_fp32.h"
#include "nnacl/errorcode.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BiasAdd;

namespace mindspore::kernel {
int BiasAddInt8CPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int BiasAddInt8CPUKernel::ReSize() {
  auto bias_param = reinterpret_cast<ArithmeticParameter *>(op_parameter_);
  auto dims = in_tensors_.at(0)->shape();
  bias_param->ndim_ = dims.size();
  if (bias_param->ndim_ < 1 || bias_param->ndim_ > 5) {
    MS_LOG(ERROR) << "input shape is invalid";
    return RET_ERROR;
  }
  for (size_t i = 0; i < bias_param->ndim_; i++) {
    bias_param->in_shape0_[i] = dims[i];
    bias_param->in_shape1_[i] = 1;
    bias_param->out_shape_[i] = dims[i];
  }
  bias_param->in_shape1_[bias_param->ndim_ - 1] = dims[bias_param->ndim_ - 1];
  return RET_OK;
}

int BiasAddInt8CPUKernel::Run() {
  auto in = reinterpret_cast<int8_t *>(in_tensors_.at(0)->MutableData());
  auto bias = reinterpret_cast<int8_t *>(in_tensors_.at(1)->MutableData());
  auto out = reinterpret_cast<int8_t *>(out_tensors_.at(0)->MutableData());
  size_t data_size = in_tensors_.at(0)->ElementsNum();
  auto tile_in = static_cast<int8_t *>(ctx_->allocator->Malloc(data_size));
  if (tile_in == nullptr) {
    MS_LOG(ERROR) << "Failed to malloc momery";
    return NNACL_ERR;
  }
  auto tile_bias = static_cast<int8_t *>(ctx_->allocator->Malloc(data_size));
  if (tile_bias == nullptr) {
    MS_LOG(ERROR) << "Failed to malloc momery";
    ctx_->allocator->Free(tile_in);
    return NNACL_ERR;
  }
  BroadcastAddInt8(in, bias, tile_in, tile_bias, out, data_size,
                   reinterpret_cast<ArithmeticParameter *>(op_parameter_));
  ctx_->allocator->Free(tile_in);
  ctx_->allocator->Free(tile_bias);
  return NNACL_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_BiasAdd, LiteKernelCreator<BiasAddInt8CPUKernel>)
}  // namespace mindspore::kernel
