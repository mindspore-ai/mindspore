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

#include "src/runtime/kernel/arm/base/tile_base.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_TileFusion;

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

  auto data_type = in_tensors_.at(0)->data_type();
  if (data_type == kNumberTypeFloat32 || data_type == kNumberTypeInt32) {
    tile_parameter_->data_size_ = sizeof(float);
  } else if (data_type == kNumberTypeFloat16) {
    tile_parameter_->data_size_ = sizeof(float) / 2;
  } else {
    MS_LOG(ERROR) << "tile not support data type: " << data_type;
    return RET_ERROR;
  }

  FillOneDimTileParam();
  return RET_OK;
}

int SimpleTile(void *cdata, int task_id) {
  auto kernel = reinterpret_cast<TileCPUKernel *>(cdata);
  auto ret = kernel->SimpleTileImpl(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SimpleTile error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

void TileCPUKernel::FillOneDimTileParam() {
  // check if tile exact one dim
  auto tile_parameter_ = reinterpret_cast<TileParameter *>(op_parameter_);
  MS_ASSERT(tile_parameter_);
  int large_one_multiple_count = 0;
  int multiple;
  int mul_index;
  for (auto i = 0; i < tile_parameter_->in_dim_; ++i) {
    if (tile_parameter_->multiples_[i] > 1) {
      large_one_multiple_count++;
      multiple = tile_parameter_->multiples_[i];
      mul_index = i;
    }
  }
  one_dim_tile_ = large_one_multiple_count == 1;
  if (one_dim_tile_) {
    tile_parameter_->fast_multiple_ = static_cast<size_t>(multiple);
    tile_parameter_->fast_stride_ =
      static_cast<size_t>(tile_parameter_->in_shape_[mul_index] * tile_parameter_->in_strides_[mul_index]);
    tile_parameter_->fast_outer_size_ =
      static_cast<size_t>(in_tensors_.at(0)->ElementsNum()) / tile_parameter_->fast_stride_;
  }
  return;
}

int TileCPUKernel::SimpleTileImpl(int task_id) {
  auto param = reinterpret_cast<TileParameter *>(op_parameter_);
  MS_ASSERT(param);
  size_t unit = UP_DIV(param->fast_outer_size_, static_cast<size_t>(context_->thread_num_));
  if (unit == 0 && task_id > 0) {
    return RET_OK;
  }
  size_t begin = unit * static_cast<size_t>(task_id);
  size_t end = MSMIN(begin + unit, param->fast_outer_size_);
  TileSimple(input_addr_, output_addr_, begin, end, param);
  return RET_OK;
}

int TileCPUKernel::RunSimpleTile() {
  auto ret = ParallelLaunch(context_->thread_pool_, SimpleTile, this, context_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "RunSimpleTile error code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int TileCPUKernel::Run() {
  input_addr_ = reinterpret_cast<uint8_t *>(in_tensors_.at(0)->data_c());
  output_addr_ = reinterpret_cast<uint8_t *>(out_tensors_.at(0)->data_c());
  MS_ASSERT(input_addr_ != nullptr);
  MS_ASSERT(output_addr_ != nullptr);
  if (one_dim_tile_) {
    return RunSimpleTile();
  }
  Tile(input_addr_, output_addr_, reinterpret_cast<TileParameter *>(op_parameter_));
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_TileFusion, LiteKernelCreator<TileCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_TileFusion, LiteKernelCreator<TileCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_TileFusion, LiteKernelCreator<TileCPUKernel>)
}  // namespace mindspore::kernel
