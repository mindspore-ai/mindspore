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

#include "nnacl/kernel/batch_to_space.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/base/batch_to_space_base.h"
#include "nnacl/nnacl_common.h"
#include "nnacl/tensor_c_utils.h"
#include "nnacl/batch_to_space_parameter.h"

int BatchToSpaceProcessInput(BatchToSpaceStruct *batch_to_space) {
  TensorC *block_shape = batch_to_space->base_.in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(block_shape);
  NNACL_CHECK_NULL_RETURN_ERR(block_shape->data_);
  TensorC *crop = batch_to_space->base_.in_[THIRD_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(crop);
  NNACL_CHECK_NULL_RETURN_ERR(crop->data_);

  if (GetElementNum(block_shape) < BATCH_TO_SPACE_BLOCK_SHAPE_SIZE) {
    return NNACL_BATCH_TO_SPACE_BLOCK_SHAPE_INVALID;
  }
  if (GetElementNum(crop) < COMM_SHAPE_SIZE) {
    return NNACL_BATCH_TO_SPACE_CROP_INVALID;
  }

  int32_t *block_shape_data = (int32_t *)block_shape->data_;
  for (int i = 0; i < BATCH_TO_SPACE_BLOCK_SHAPE_SIZE; ++i) {
    batch_to_space->block_shape_[i] = block_shape_data[i];
  }

  int32_t *crops_data = (int32_t *)crop->data_;
  batch_to_space->no_crop_ = true;
  for (int i = 0; i < COMM_SHAPE_SIZE; ++i) {
    batch_to_space->crops_[i] = crops_data[i];
    if (batch_to_space->crops_[i] != 0) {
      batch_to_space->no_crop_ = false;
    }
  }
  return NNACL_OK;
}

int BatchToSpaceCompute(KernelBase *self) {
  BatchToSpaceStruct *batch_to_space = (BatchToSpaceStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(batch_to_space);
  TensorC *input = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input);
  TensorC *output = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output);

  size_t data_size = DataTypeCSize(input->data_type_);
  if (self->in_size_ == Num1) {
    if (batch_to_space->no_crop_) {
      BatchToSpaceNoCropForNHWC(input->data_, output->data_, input->shape_, output->shape_[Index0],
                                batch_to_space->block_shape_, data_size);
    } else {
      BatchToSpaceForNHWC(input->data_, output->data_, input->shape_, output->shape_[Index0],
                          batch_to_space->block_shape_, batch_to_space->crops_, data_size);
    }
  }

  if (self->in_size_ == Num3) {
    int ret = BatchToSpaceProcessInput(batch_to_space);
    if (ret != NNACL_OK) {
      return ret;
    }
    if (batch_to_space->no_crop_) {
      BatchToSpaceNoCropForNHWC(input->data_, output->data_, input->shape_, output->shape_[Index0],
                                batch_to_space->block_shape_, data_size);
    } else {
      BatchToSpaceForNHWC(input->data_, output->data_, input->shape_, output->shape_[Index0],
                          batch_to_space->block_shape_, batch_to_space->crops_, data_size);
    }
  }
  return NNACL_OK;
}

int BatchToSpaceResize(KernelBase *self) {
  NNACL_CHECK_NULL_RETURN_ERR(self->in_[FIRST_INPUT]);
  NNACL_CHECK_NULL_RETURN_ERR(self->out_[OUTPUT_INDEX]);
  NNACL_CHECK_TRUE_RET(self->in_[FIRST_INPUT]->shape_size_ == COMM_SHAPE_SIZE, NNACL_ERR);
  return NNACL_OK;
}

KernelBase *CreateBatchToSpace(OpParameter *param, int data_type) {
  BatchToSpaceStruct *batch_to_space = (BatchToSpaceStruct *)malloc(sizeof(BatchToSpaceStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(batch_to_space);
  memset(batch_to_space, 0, sizeof(BatchToSpaceStruct));
  BatchToSpaceParameter *bts_param = (BatchToSpaceParameter *)param;
  NNACL_CHECK_NULL_RETURN_NULL(bts_param);
  memcpy(batch_to_space->crops_, bts_param->crops_, sizeof(int32_t) * COMM_SHAPE_SIZE);
  memcpy(batch_to_space->block_shape_, bts_param->block_shape_, sizeof(int32_t) * BATCH_TO_SPACE_BLOCK_SHAPE_SIZE);
  batch_to_space->base_.Prepare = DefaultPrepare1In1Out;
  batch_to_space->base_.Resize = BatchToSpaceResize;
  batch_to_space->base_.Release = DefaultRelease;
  batch_to_space->base_.Compute = BatchToSpaceCompute;
  return (KernelBase *)batch_to_space;
}

REG_KERNEL_CREATOR(PrimType_BatchToSpace, kNumberTypeFloat16, CreateBatchToSpace)
REG_KERNEL_CREATOR(PrimType_BatchToSpace, kNumberTypeFloat32, CreateBatchToSpace)
REG_KERNEL_CREATOR(PrimType_BatchToSpaceND, kNumberTypeFloat16, CreateBatchToSpace)
REG_KERNEL_CREATOR(PrimType_BatchToSpaceND, kNumberTypeFloat32, CreateBatchToSpace)
