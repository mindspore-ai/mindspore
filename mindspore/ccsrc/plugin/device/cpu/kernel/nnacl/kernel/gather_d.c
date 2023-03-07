/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either gather_dress or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nnacl/kernel/gather_d.h"
#include "nnacl/gather_parameter.h"
#include "nnacl/tensor_c_utils.h"
#include "nnacl/op_base.h"
#include "nnacl/base/gather_d_base.h"

typedef struct GatherDStru {
  KernelBase base;
} GatherDStru;

int gather_d_prepare(struct KernelBase *self) {
  GatherDStru *gather_d = (GatherDStru *)self;
  NNACL_CHECK_NULL_RETURN_ERR(gather_d);
  GatherParameter *param = (GatherParameter *)gather_d->base.param_;
  NNACL_CHECK_NULL_RETURN_ERR(param);
  if (self->in_size_ < kInputSize2 || self->out_size_ < 1) {
    return NNACL_ERR;
  }
  param->axis_ = ((int *)(gather_d->base.in_[1].data_))[0];
  return NNACL_OK;
}

int gather_d_resize(struct KernelBase *self) {
  GatherDStru *gather_d = (GatherDStru *)self;
  NNACL_CHECK_NULL_RETURN_ERR(gather_d);
  GatherParameter *param = (GatherParameter *)gather_d->base.param_;
  NNACL_CHECK_NULL_RETURN_ERR(param);
  int input_rank = gather_d->base.in_[0].shape_size_;
  if (param->axis_ >= input_rank || param->axis_ < -input_rank) {
    return NNACL_PARAM_INVALID;
  }
  if (param->axis_ < 0) {
    param->axis_ = param->axis_ + input_rank;
  }
  return NNACL_OK;
}

int gather_d_release(struct KernelBase *self) { return NNACL_OK; }

int gather_d_compute(struct KernelBase *self) {
  GatherDStru *gather_d_stru = (GatherDStru *)self;
  NNACL_CHECK_NULL_RETURN_ERR(gather_d_stru);
  GatherParameter *param = (GatherParameter *)gather_d_stru->base.param_;
  NNACL_CHECK_NULL_RETURN_ERR(param);

  TensorC *input = &(gather_d_stru->base.in_[0]);
  NNACL_CHECK_NULL_RETURN_ERR(input);
  TensorC *output = &(gather_d_stru->base.out_[0]);
  NNACL_CHECK_NULL_RETURN_ERR(output);
  const void *input_data = input->data_;
  NNACL_CHECK_NULL_RETURN_ERR(input_data);
  const void *index_data = gather_d_stru->base.in_[2].data_;
  NNACL_CHECK_NULL_RETURN_ERR(index_data);
  void *output_data = output->data_;
  NNACL_CHECK_NULL_RETURN_ERR(output_data);

  size_t input_shape[MAX_SHAPE_SIZE];
  for (size_t i = 0; i < input->shape_size_; i++) {
    input_shape[i] = input->shape_[i];
  }
  size_t output_shape[MAX_SHAPE_SIZE];
  for (size_t i = 0; i < output->shape_size_; i++) {
    output_shape[i] = output->shape_[i];
  }

  TypeIdC input_dtype = input->data_type_;
  TypeIdC index_dtype = gather_d_stru->base.in_[2].data_type_;
  int status = NNACL_ERR;
  if (index_dtype == kNumberTypeInt32) {
    if (input_dtype == kNumberTypeFloat32) {
      status = GATHER_D(float, int32_t, (float *)output_data, (float *)input_data, (int32_t *)index_data, input_shape,
                        input->shape_size_, output_shape, output->shape_size_, param->axis_);
    } else if (input_dtype == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
      status = GATHER_D(float16_t, int32_t, (float16_t *)output_data, (float16_t *)input_data, (int32_t *)index_data,
                        input_shape, input->shape_size_, output_shape, output->shape_size_, param->axis_);
#endif
    } else if (input_dtype == kNumberTypeInt32) {
      status = GATHER_D(int32_t, int32_t, (int32_t *)output_data, (int32_t *)input_data, (int32_t *)index_data,
                        input_shape, input->shape_size_, output_shape, output->shape_size_, param->axis_);
    }
  } else if (index_dtype == kNumberTypeInt64) {
    if (input_dtype == kNumberTypeFloat32) {
      status = GATHER_D(float, int64_t, (float *)output_data, (float *)input_data, (int64_t *)index_data, input_shape,
                        input->shape_size_, output_shape, output->shape_size_, param->axis_);
    } else if (input_dtype == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
      status = GATHER_D(float16_t, int64_t, (float16_t *)output_data, (float16_t *)input_data, (int64_t *)index_data,
                        input_shape, input->shape_size_, output_shape, output->shape_size_, param->axis_);
#endif
    } else if (input_dtype == kNumberTypeInt32) {
      status = GATHER_D(int32_t, int64_t, (int32_t *)output_data, (int32_t *)input_data, (int64_t *)index_data,
                        input_shape, input->shape_size_, output_shape, output->shape_size_, param->axis_);
    }
  }
  return status;
}

KernelBase *CreateGatherD(OpParameter *param, int data_type) {
  GatherDStru *gather_d = (GatherDStru *)malloc(sizeof(GatherDStru));
  NNACL_CHECK_NULL_RETURN_NULL(gather_d);
  gather_d->base.prepare = gather_d_prepare;
  gather_d->base.resize = gather_d_resize;
  gather_d->base.release = gather_d_release;
  gather_d->base.compute = gather_d_compute;
  return (KernelBase *)gather_d;
}

REG_KERNEL_CREATOR(PrimType_GatherD, kNumberTypeFloat32, CreateGatherD);
REG_KERNEL_CREATOR(PrimType_GatherD, kNumberTypeInt32, CreateGatherD);
REG_KERNEL_CREATOR(PrimType_GatherD, kNumberTypeFloat16, CreateGatherD);
