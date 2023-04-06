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

#include "nnacl/kernel/biasadd.h"
#include "nnacl/tensor_c.h"
#include "nnacl/op_base.h"
#include "nnacl/fp32/bias_add.h"

#define BIAS_ADD_PER_UNIT_LOAD_NUM 2
#define BIAS_ADD_PER_UNIT_STORE_NUM 1

typedef struct BiasAddStruct {
  KernelBase base_;
  int64_t inner_num_;
  int64_t outer_num_;
  int64_t total_num_;
  bool batch_priority_;
  int64_t *split_points_;
  int split_pionts_size_;
} BiasAddStruct;

int biasadd_release(struct KernelBase *self) {
  BiasAddStruct *bias_add = (BiasAddStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(bias_add);
  if (bias_add->split_points_ != NULL) {
    bias_add->base_.env_->free(bias_add->base_.env_->allocator_, bias_add->split_points_);
  }
  return NNACL_OK;
}

int ChooseBiasThreadCuttingStrategy(KernelBase *self) {
  BiasAddStruct *bias_add = (BiasAddStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(bias_add);
  self->thread_nr_ = self->update_thread_(TC_PTYPE(PrimType_BiasAdd), BIAS_ADD_PER_UNIT_LOAD_NUM,
                                          BIAS_ADD_PER_UNIT_STORE_NUM, bias_add->total_num_, self->thread_nr_);

  int64_t tmp_split_list[MAX_SPLIT_NUM];
  bias_add->split_pionts_size_ = 0;
  int64_t block_size = 1;
  block_size = bias_add->total_num_ / self->thread_nr_;
  int64_t remain_data = bias_add->total_num_ - block_size * self->thread_nr_;
  int64_t split_point = 0;
  while (split_point < bias_add->total_num_) {
    tmp_split_list[bias_add->split_pionts_size_++] = split_point;
    split_point += block_size;
    if (remain_data > 0) {
      ++split_point;
      --remain_data;
    }
  }

  self->thread_nr_ = bias_add->split_pionts_size_;
  biasadd_release(self);
  bias_add->split_points_ = self->env_->alloc(self->env_->allocator_, bias_add->split_pionts_size_ * sizeof(int64_t));
  NNACL_CHECK_NULL_RETURN_ERR(bias_add->split_points_);
  memcpy(bias_add->split_points_, tmp_split_list, bias_add->split_pionts_size_ * sizeof(int64_t));

  if (bias_add->inner_num_ >= C64NUM && block_size / bias_add->inner_num_ >= C6NUM) {
    bias_add->batch_priority_ = true;
  } else {
    bias_add->batch_priority_ = false;
  }
  return NNACL_OK;
}

int BiasRun(void *cdata, int task_id, float l, float r) {
  BiasAddStruct *bias_add = (BiasAddStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(bias_add);

  float *input = (float *)(bias_add->base_.in_[FIRST_INPUT]->data_);
  NNACL_CHECK_NULL_RETURN_ERR(input);
  float *bias = (float *)(bias_add->base_.in_[SECOND_INPUT]->data_);
  NNACL_CHECK_NULL_RETURN_ERR(bias);
  float *output = (float *)(bias_add->base_.out_[FIRST_INPUT]->data_);
  NNACL_CHECK_NULL_RETURN_ERR(output);

  int64_t block_start = bias_add->split_points_[task_id];
  int64_t block_end = bias_add->total_num_;
  if ((task_id + 1) < bias_add->split_pionts_size_) {
    block_end = bias_add->split_points_[task_id + 1];
  }
  BiasAddOpt(input, bias, output, block_start, block_end, bias_add->inner_num_, bias_add->batch_priority_);
  return NNACL_OK;
}

int biasadd_prepare(struct KernelBase *self) {
  MS_CHECK_FALSE(self->in_size_ < C2NUM, NNACL_ERR);
  MS_CHECK_FALSE(self->out_size_ < C1NUM, NNACL_ERR);
  return NNACL_OK;
}

int biasadd_resize(struct KernelBase *self) {
  BiasAddStruct *bias_add = (BiasAddStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(bias_add);

  TensorC *in_tensor = self->in_[FIRST_INPUT];
  TensorC *add_tensor = self->in_[SECOND_INPUT];
  MS_CHECK_FALSE(in_tensor->shape_size_ == 0, NNACL_ERR);
  MS_CHECK_FALSE(add_tensor->shape_size_ == 0, NNACL_ERR);
  MS_CHECK_FALSE(in_tensor->shape_size_ < add_tensor->shape_size_, NNACL_ERR);

  size_t dim_offset = in_tensor->shape_size_ - add_tensor->shape_size_;
  bias_add->inner_num_ = 1;
  for (size_t i = 0; i < add_tensor->shape_size_; ++i) {
    MS_CHECK_FALSE(in_tensor->shape_[i + dim_offset] != add_tensor->shape_[i], NNACL_BIAS_ADD_SHAPE_NOT_MATCH);
    MS_CHECK_FALSE(INT_MUL_OVERFLOW(in_tensor->shape_[i], bias_add->inner_num_), NNACL_BIAS_ADD_SHAPE_OVERFLOW);
    bias_add->inner_num_ *= add_tensor->shape_[i];
  }

  bias_add->outer_num_ = 1;
  for (size_t i = 0; i < dim_offset; ++i) {
    MS_CHECK_FALSE(INT_MUL_OVERFLOW(in_tensor->shape_[i], bias_add->outer_num_), NNACL_BIAS_ADD_SHAPE_OVERFLOW);
    bias_add->outer_num_ *= in_tensor->shape_[i];
  }

  MS_CHECK_FALSE(INT_MUL_OVERFLOW(bias_add->inner_num_, bias_add->outer_num_), NNACL_BIAS_ADD_SHAPE_OVERFLOW);
  bias_add->total_num_ = bias_add->inner_num_ * bias_add->outer_num_;
  return ChooseBiasThreadCuttingStrategy(self);
}

int biasadd_compute(struct KernelBase *self) {
  return self->env_->parallel_launch(self->env_->thread_pool_, BiasRun, self, self->thread_nr_);
}

KernelBase *CreateBiasAdd(OpParameter *param, int data_type) {
  BiasAddStruct *bias_add = (BiasAddStruct *)malloc(sizeof(BiasAddStruct));
  NNACL_CHECK_NULL_RETURN_NULL(bias_add);
  bias_add->split_points_ = NULL;
  bias_add->base_.prepare = biasadd_prepare;
  bias_add->base_.resize = biasadd_resize;
  bias_add->base_.release = biasadd_release;
  bias_add->base_.compute = biasadd_compute;
  return (KernelBase *)bias_add;
}

REG_KERNEL_CREATOR(PrimType_BiasAdd, kNumberTypeFloat32, CreateBiasAdd);
REG_KERNEL_CREATOR(PrimType_BiasAdd, kNumberTypeFloat16, CreateBiasAdd);
