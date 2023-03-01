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

#include "src/litert/kernel/cpu/base/tile_base.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/nnacl_common.h"
#include "include/errorcode.h"
#include "nnacl/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_TileFusion;

namespace mindspore::kernel {
namespace {
constexpr size_t kDoubleInputsSize = 2;
}
int TileCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int TileCPUKernel::DoubleInputScenes() {
  CHECK_NULL_RETURN(in_tensors_.at(1));
  if (in_tensors_[1]->data() == nullptr) {
    resize_done_ = false;
    return RET_OK;
  }
  if (in_tensors_[1]->ElementsNum() > static_cast<int>(in_tensors_[0]->shape().size())) {
    MS_LOG(ERROR) << "tile's input1 data_num cannot be larger than input0's shape_size.";
    return RET_ERROR;
  }
  if (in_tensors_[1]->data_type() != kNumberTypeInt && in_tensors_[1]->data_type() != kNumberTypeInt32) {
    MS_LOG(ERROR) << "in_tensors_[1]->data_type():" << in_tensors_[1]->data_type()
                  << " must be kNumberTypeInt32 or kNumberTypeInt!";
    return RET_ERROR;
  }
  CHECK_NULL_RETURN(in_tensors_[1]->data());
  auto input1_addr = reinterpret_cast<int *>(in_tensors_[1]->data());
  for (int i = 0; i < in_tensors_[1]->ElementsNum(); ++i) {
    if (input1_addr[i] <= 0) {
      MS_LOG(ERROR) << "Tile input1 data must be greater than 0";
      return RET_ERROR;
    }
    tile_parameter_->dims_[i] = i;
    tile_parameter_->multiples_[i] = input1_addr[i];
  }
  return RET_OK;
}

int TileCPUKernel::ReSize() {
  auto ret = RET_OK;

  CHECK_NULL_RETURN(tile_parameter_);
  if (in_tensors_.size() == kDoubleInputsSize) {
    ret = DoubleInputScenes();
    if (ret != RET_OK) {
      return ret;
    }
  }

  CHECK_NULL_RETURN(in_tensors_.at(0));
  CHECK_NULL_RETURN(out_tensors_.at(0));
  tile_parameter_->in_dim_ = in_tensors_.at(0)->shape().size();
  MS_CHECK_TRUE_RET(tile_parameter_->in_dim_ > 0 && tile_parameter_->in_dim_ <= MAX_TILE_DIM_SIZE, RET_ERROR);
  CHECK_LESS_RETURN((int)(out_tensors_.at(0)->shape().size()), tile_parameter_->in_dim_);
  for (int i = 0; i < tile_parameter_->in_dim_; ++i) {
    tile_parameter_->in_shape_[i] = in_tensors_.at(0)->shape().at(i);
    tile_parameter_->out_shape_[i] = out_tensors_.at(0)->shape().at(i);
  }
  ComputeStrides(tile_parameter_->in_shape_, tile_parameter_->in_strides_, tile_parameter_->in_dim_);
  ComputeStrides(tile_parameter_->out_shape_, tile_parameter_->out_strides_, tile_parameter_->in_dim_);

  for (size_t i = 0; i < tile_parameter_->dims_size_; i++) {
    MS_CHECK_FALSE(INT_MUL_OVERFLOW(tile_parameter_->multiples_[i], tile_parameter_->in_shape_[i]),
                   NNACL_ERRCODE_MUL_OVERFLOW);
    auto ele_num = tile_parameter_->multiples_[i] * tile_parameter_->in_shape_[i] - 1;
    MS_CHECK_FALSE(INT_MUL_OVERFLOW(tile_parameter_->out_strides_[i], ele_num), NNACL_ERRCODE_MUL_OVERFLOW);
  }

  auto data_type = in_tensors_.at(0)->data_type();
  if (data_type == kNumberTypeFloat32 || data_type == kNumberTypeInt32) {
    tile_parameter_->data_size_ = sizeof(float);
  } else if (data_type == kNumberTypeFloat16) {
    tile_parameter_->data_size_ = sizeof(float) / 2;
  } else if (data_type == kNumberTypeBool) {
    tile_parameter_->data_size_ = sizeof(bool);
  } else {
    MS_LOG(ERROR) << "tile not support data type: " << data_type;
    return RET_ERROR;
  }

  ret = FillOneDimTileParam();
  if (ret != RET_OK) {
    return ret;
  }

  if (one_dim_tile_) {
    if (UpdateThreadNumPass(TC_TYPE(schema::PrimitiveType_TileFusion, 0), 0, 0, tile_parameter_->fast_outer_size_) !=
        RET_OK) {
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int SimpleTile(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto kernel = reinterpret_cast<TileCPUKernel *>(cdata);
  CHECK_NULL_RETURN(kernel);
  auto ret = kernel->SimpleTileImpl(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SimpleTile error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int TileCPUKernel::FillOneDimTileParam() {
  // check if tile exact one dim
  int large_one_multiple_count = 0;
  int multiple = 0;
  int mul_index = 0;
  CHECK_LESS_RETURN(MAX_TILE_DIM_SIZE - 1, tile_parameter_->in_dim_);
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
    MS_CHECK_FALSE(INT_MUL_OVERFLOW(tile_parameter_->in_shape_[mul_index], tile_parameter_->in_strides_[mul_index]),
                   mindspore::lite::RET_ERROR);
    tile_parameter_->fast_stride_ =
      static_cast<size_t>(tile_parameter_->in_shape_[mul_index] * tile_parameter_->in_strides_[mul_index]);
    CHECK_LESS_RETURN(tile_parameter_->fast_stride_, 1);
    tile_parameter_->fast_outer_size_ =
      static_cast<size_t>(in_tensors_.at(0)->ElementsNum()) / tile_parameter_->fast_stride_;
  }
  resize_done_ = true;
  return RET_OK;
}

int TileCPUKernel::SimpleTileImpl(int task_id) {
  CHECK_LESS_RETURN(static_cast<size_t>(thread_num_), 1);
  size_t unit = UP_DIV(tile_parameter_->fast_outer_size_, static_cast<size_t>(thread_num_));
  if (unit == 0 && task_id > 0) {
    return RET_OK;
  }
  MS_CHECK_FALSE(INT_MUL_OVERFLOW(unit, static_cast<size_t>(task_id)), RET_ERROR);
  size_t begin = unit * static_cast<size_t>(task_id);
  size_t end = MSMIN(begin + unit, tile_parameter_->fast_outer_size_);
  TileSimple(input_addr_, output_addr_, begin, end, tile_parameter_);
  return RET_OK;
}

int TileCPUKernel::RunSimpleTile() {
  auto ret = ParallelLaunch(this->ms_context_, SimpleTile, this, thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "RunSimpleTile error code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int TileCPUKernel::Run() {
  auto data_type = in_tensors_.at(0)->data_type();
  tile_parameter_->data_size_ = lite::DataTypeSize(data_type);
  input_addr_ = reinterpret_cast<uint8_t *>(in_tensors_.at(0)->data());
  output_addr_ = reinterpret_cast<uint8_t *>(out_tensors_.at(0)->data());
  CHECK_NULL_RETURN(input_addr_);
  CHECK_NULL_RETURN(output_addr_);
  if (!resize_done_) {
    int ret = ReSize();
    if (ret != RET_OK || !resize_done_) {
      MS_LOG(ERROR) << "Tile Resize error.";
      return ret;
    }
  }
  if (one_dim_tile_) {
    return RunSimpleTile();
  }
  Tile(input_addr_, output_addr_, reinterpret_cast<TileParameter *>(op_parameter_));
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_TileFusion, LiteKernelCreator<TileCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_TileFusion, LiteKernelCreator<TileCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_TileFusion, LiteKernelCreator<TileCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_TileFusion, LiteKernelCreator<TileCPUKernel>)
}  // namespace mindspore::kernel
