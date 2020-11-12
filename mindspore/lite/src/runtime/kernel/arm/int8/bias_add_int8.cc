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
#include "nnacl/fp32/arithmetic.h"
#include "nnacl/errorcode.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
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
  auto dims = in_tensors_[0]->shape();
  bias_param->ndim_ = dims.size();
  for (size_t i = 0; i < bias_param->ndim_; i++) {
    bias_param->in_shape0_[i] = dims[i];
    bias_param->in_shape1_[i] = 1;
    bias_param->out_shape_[i] = dims[i];
  }
  bias_param->in_shape1_[3] = dims[3];
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

kernel::LiteKernel *CpuBiasAddInt8KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                const std::vector<lite::Tensor *> &outputs, OpParameter *parameter,
                                                const lite::InnerContext *ctx, const KernelKey &desc,
                                                const mindspore::lite::PrimitiveC *primitive) {
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "parameter is nullptr";
    return nullptr;
  }
  if (ctx == nullptr) {
    MS_LOG(ERROR) << "ctx is nullptr";
    free(parameter);
    return nullptr;
  }
  MS_ASSERT(desc.type == PrimitiveType_BiasAdd);
  auto *kernel = new (std::nothrow) BiasAddInt8CPUKernel(parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create kernel failed, name: " << parameter->name_;
    free(parameter);
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << parameter->name_
                  << ", type: " << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(parameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_BiasAdd, CpuBiasAddInt8KernelCreator)
}  // namespace mindspore::kernel
