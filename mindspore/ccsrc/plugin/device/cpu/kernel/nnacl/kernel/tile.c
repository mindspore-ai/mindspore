/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "nnacl/kernel/tile.h"
#include "nnacl/tile_parameter.h"
#include "nnacl/tensor_c_utils.h"
#include "nnacl/nnacl_common.h"
#include "nnacl/op_base.h"
#include "nnacl/base/tile_base.h"
#include "nnacl/kernel/default_kernel_base.h"

#define kDoubleInputsSize 2

int TileDoubleInputScenes(TileStruct *tile) {
  TensorC *t = tile->base_.in_[SECOND_INPUT];
  if (t->data_ == NULL) {
    tile->resize_done_ = false;
    return NNACL_OK;
  }

  NNACL_CHECK_FALSE(GetElementNum(t) > (int)tile->base_.in_[FIRST_INPUT]->shape_size_,
                    NNACL_TILE_SECOND_INPUT_NUM_INVALID);
  NNACL_CHECK_FALSE(t->data_type_ != kNumberTypeInt && t->data_type_ != kNumberTypeInt32,
                    NNACL_TILE_SECOND_INPUT_DATA_TYPE_INVALID);

  int *input1_addr = (int *)(t->data_);
  for (int i = 0; i < GetElementNum(t); ++i) {
    NNACL_CHECK_FALSE(input1_addr[i] <= 0, NNACL_TILE_SECOND_INPUT_VALUE_INVALID);
    tile->dims_[i] = i;
    tile->multiples_[i] = input1_addr[i];
  }
  return NNACL_OK;
}

int SimpleTileImpl(TileStruct *tile, int task_id) {
  NNACL_CHECK_ZERO_RETURN_ERR(tile->base_.thread_nr_);
  size_t unit = UP_DIV(tile->fast_outer_size_, (size_t)tile->base_.thread_nr_);
  if (unit == 0 && task_id > 0) {
    return NNACL_OK;
  }
  NNACL_CHECK_FALSE(INT_MUL_OVERFLOW(unit, (size_t)task_id), NNACL_ERR);
  size_t begin = unit * (size_t)(task_id);
  size_t end = MSMIN(begin + unit, tile->fast_outer_size_);
  TileSimple(tile->input_addr_, tile->output_addr_, begin, end, tile);
  return NNACL_OK;
}

int SimpleTile(void *cdata, int task_id, float l, float r) {
  TileStruct *tile = (TileStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(tile);
  return SimpleTileImpl(tile, task_id);
}

int TileFillOneDimTileParam(TileStruct *tile) {
  // check if tile exact one dim
  int large_one_multiple_count = 0;
  int multiple = 0;
  int mul_index = 0;

  for (int i = 0; i < tile->in_dim_; ++i) {
    if (tile->multiples_[i] > 1) {
      large_one_multiple_count++;
      multiple = tile->multiples_[i];
      mul_index = i;
    }
  }
  tile->one_dim_tile_ = large_one_multiple_count == 1;
  if (tile->one_dim_tile_) {
    tile->fast_multiple_ = (size_t)multiple;
    NNACL_CHECK_FALSE(INT_MUL_OVERFLOW(tile->in_shape_[mul_index], tile->in_strides_[mul_index]), NNACL_ERR);
    tile->fast_stride_ = (size_t)(tile->in_shape_[mul_index] * tile->in_strides_[mul_index]);
    NNACL_CHECK_FALSE(tile->fast_stride_ < 1, NNACL_TILE_INPUT_SHAPE_INVALID);
    tile->fast_outer_size_ = (size_t)GetElementNum(tile->base_.in_[FIRST_INPUT]) / tile->fast_stride_;
  }
  tile->resize_done_ = true;
  return NNACL_OK;
}

int TileResize(struct KernelBase *self) {
  TileStruct *tile = (TileStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(tile);
  TileParameter *param = (TileParameter *)(self->param_);
  NNACL_CHECK_NULL_RETURN_ERR(tile);

  tile->dims_size_ = param->dims_size_;
  for (int i = 0; i < MAX_SHAPE_SIZE; i++) {
    tile->dims_[i] = param->dims_[i];
    tile->multiples_[i] = param->multiples_[i];
  }

  if (self->in_size_ == kDoubleInputsSize) {
    int ret = TileDoubleInputScenes(tile);
    NNACL_CHECK_FALSE(ret != NNACL_OK, ret);
  }

  TensorC *input = self->in_[0];
  TensorC *output = self->out_[0];
  NNACL_CHECK_NULL_RETURN_ERR(input);
  NNACL_CHECK_NULL_RETURN_ERR(output);

  tile->in_dim_ = (int)input->shape_size_;
  NNACL_CHECK_TRUE_RET(tile->in_dim_ > 0 && tile->in_dim_ <= MAX_SHAPE_SIZE, NNACL_TILE_INPUT_SHAPE_INVALID);
  NNACL_CHECK_FALSE((int)output->shape_size_ < tile->in_dim_, NNACL_TILE_INPUT_SHAPE_INVALID);

  for (int i = 0; i < tile->in_dim_; ++i) {
    tile->in_shape_[i] = input->shape_[i];
    tile->out_shape_[i] = output->shape_[i];
  }

  ComputeStrides(tile->in_shape_, tile->in_strides_, tile->in_dim_);
  ComputeStrides(tile->out_shape_, tile->out_strides_, tile->in_dim_);

  for (size_t i = 0; i < tile->dims_size_; i++) {
    NNACL_CHECK_FALSE(INT_MUL_OVERFLOW(tile->multiples_[i], tile->in_shape_[i]), NNACL_ERRCODE_MUL_OVERFLOW);
    int ele_num = tile->multiples_[i] * tile->in_shape_[i] - 1;
    NNACL_CHECK_FALSE(INT_MUL_OVERFLOW(tile->out_strides_[i], ele_num), NNACL_ERRCODE_MUL_OVERFLOW);
  }

  int ret = TileFillOneDimTileParam(tile);
  NNACL_CHECK_FALSE(ret != NNACL_OK, ret);

  if (tile->one_dim_tile_) {
    self->thread_nr_ =
      self->UpdateThread(TC_TYPE(PrimType_TileFusion, 0), 0, 0, tile->fast_outer_size_, self->thread_nr_);
  }
  return NNACL_OK;
}

int TileCompute(struct KernelBase *self) {
  TileStruct *tile = (TileStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(tile);
  tile->input_addr_ = (uint8_t *)(self->in_[FIRST_INPUT]->data_);
  NNACL_CHECK_NULL_RETURN_ERR(tile->input_addr_);
  tile->output_addr_ = (uint8_t *)(self->out_[OUTPUT_INDEX]->data_);
  NNACL_CHECK_NULL_RETURN_ERR(tile->output_addr_);

  if (!tile->resize_done_) {
    int ret = TileResize(self);
    NNACL_CHECK_FALSE(ret != NNACL_OK, ret);
    NNACL_CHECK_FALSE(tile->resize_done_ == false, NNACL_TILE_RESIZE_IN_RUNTIME_FAILED);
  }

  tile->data_size_ = DataTypeCSize(self->in_[FIRST_INPUT]->data_type_);
  NNACL_CHECK_TRUE_RET(tile->data_size_ > 0, NNACL_UNSUPPORTED_DATA_TYPE);

  if (tile->one_dim_tile_) {
    return self->env_->ParallelLaunch(self->env_->thread_pool_, SimpleTile, self, self->thread_nr_);
  }

  Tile(tile->input_addr_, tile->output_addr_, tile);
  return NNACL_OK;
}

KernelBase *CreateTile(OpParameter *param, int data_type) {
  TileStruct *tile = (TileStruct *)malloc(sizeof(TileStruct));
  NNACL_CHECK_NULL_RETURN_NULL(tile);
  tile->resize_done_ = false;
  tile->base_.Release = DefaultRelease;
  tile->base_.Prepare = DefaultPrepare1In1Out;
  tile->base_.Resize = TileResize;
  tile->base_.Compute = TileCompute;
  return (KernelBase *)tile;
}

REG_KERNEL_CREATOR(PrimType_TileFusion, kNumberTypeInt32, CreateTile)
REG_KERNEL_CREATOR(PrimType_TileFusion, kNumberTypeFloat32, CreateTile)
REG_KERNEL_CREATOR(PrimType_TileFusion, kNumberTypeFloat16, CreateTile)
REG_KERNEL_CREATOR(PrimType_TileFusion, kNumberTypeBool, CreateTile)
REG_KERNEL_CREATOR(PrimType_TileFusion, kNumberTypeUInt8, CreateTile)
