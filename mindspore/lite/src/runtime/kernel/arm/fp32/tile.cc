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

#include "src/runtime/kernel/arm/fp32/tile.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Tile;

namespace mindspore::kernel {
int TileCPUKernel::Init() {
  auto tile_parameter_ = reinterpret_cast<TileParameter *>(opParameter);
  for (int i = 0; i < tile_parameter_->in_dim_; ++i) {
    tile_parameter_->in_shape_[i] = inputs_[0]->shape()[i];
    tile_parameter_->out_shape_[i] = outputs_[0]->shape()[i];
  }
  ComputeStrides(tile_parameter_->in_shape_, tile_parameter_->in_strides_, tile_parameter_->in_dim_);
  ComputeStrides(tile_parameter_->out_shape_, tile_parameter_->out_strides_, tile_parameter_->in_dim_);
  return RET_OK;
}

void TileCPUKernel::ComputeStrides(int *shape, int *strides, int ndim) {
  int stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
}

int TileCPUKernel::ReSize() { return RET_OK; }

int TileCPUKernel::Run() {
  auto input_addr = reinterpret_cast<float *>(inputs_.at(0)->Data());
  auto output_addr = reinterpret_cast<float *>(outputs_.at(0)->Data());

  Tile(input_addr, output_addr, reinterpret_cast<TileParameter *>(opParameter));
  return RET_OK;
}

kernel::LiteKernel *CpuTileFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                             const std::vector<lite::tensor::Tensor *> &outputs, OpParameter *parameter,
                                             const lite::Context *ctx, const KernelKey &desc) {
  if (parameter == nullptr || ctx == nullptr) {
    MS_LOG(ERROR) << "parameter or ctx is nullptr";
    return nullptr;
  }
  MS_ASSERT(desc.type == PrimitiveType_Tile);
  auto *kernel = new (std::nothrow) TileCPUKernel(parameter, inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create kernel failed, name: " << parameter->name_;
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

REG_KERNEL(kCPU, PrimitiveType_Tile, CpuTileFp32KernelCreator)
}  // namespace mindspore::kernel

