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

#include "src/runtime/kernel/arm/fp32/tile_fp32.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Tile;

namespace mindspore::kernel {
namespace {
constexpr size_t kDoubleInputsSize = 2;
}
int TileCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

void TileCPUKernel::ComputeStrides(const int *shape, int *strides, int ndim) {
  int stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
}

int TileCPUKernel::ReSize() {
  auto tile_parameter_ = reinterpret_cast<TileParameter *>(op_parameter_);
  MS_ASSERT(tile_parameter_);
  if (in_tensors_.size() == kDoubleInputsSize) {
    if (in_tensors_[1]->ElementsNum() > static_cast<int>(in_tensors_[0]->shape().size())) {
      MS_LOG(ERROR) << "tile's input1 data_num cannot be larger than input0's shape_size.";
      return false;
    }
    auto input1_addr = reinterpret_cast<int *>(in_tensors_[1]->data_c());
    for (int i = 0; i < in_tensors_[1]->ElementsNum(); ++i) {
      tile_parameter_->dims_[i] = i;
      tile_parameter_->multiples_[i] = input1_addr[i];
    }
  }
  tile_parameter_->in_dim_ = in_tensors_.at(0)->shape().size();
  for (int i = 0; i < tile_parameter_->in_dim_; ++i) {
    tile_parameter_->in_shape_[i] = in_tensors_.at(0)->shape().at(i);
    tile_parameter_->out_shape_[i] = out_tensors_.at(0)->shape().at(i);
  }
  ComputeStrides(tile_parameter_->in_shape_, tile_parameter_->in_strides_, tile_parameter_->in_dim_);
  ComputeStrides(tile_parameter_->out_shape_, tile_parameter_->out_strides_, tile_parameter_->in_dim_);
  return RET_OK;
}

int TileCPUKernel::Run() {
  auto input_addr = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  auto output_addr = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  MS_ASSERT(input_addr);
  MS_ASSERT(output_addr);
  Tile(input_addr, output_addr, reinterpret_cast<TileParameter *>(op_parameter_));
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Tile, LiteKernelCreator<TileCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Tile, LiteKernelCreator<TileCPUKernel>)
}  // namespace mindspore::kernel
