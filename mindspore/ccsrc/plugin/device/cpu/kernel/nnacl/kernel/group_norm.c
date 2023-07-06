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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nnacl/kernel/group_norm.h"
#include "nnacl/fp32/group_norm_fp32.h"
#include "nnacl/group_norm_parameter.h"
#include "nnacl/op_base.h"
#include "nnacl/errorcode.h"
#include "nnacl/tensor_c.h"
#include "nnacl/tensor_c_utils.h"

int GroupNormResize(struct KernelBase *self) {
  GroupNormStru *groupnorm = (GroupNormStru *)self;
  NNACL_CHECK_NULL_RETURN_ERR(groupnorm);
  GroupNormParameter *param = (GroupNormParameter *)groupnorm->base.param_;
  NNACL_CHECK_NULL_RETURN_ERR(param);
  NNACL_CHECK_FALSE(self->in_size_ < kInputSize2, NNACL_TENSOR_SIZE_INVALID);
  NNACL_CHECK_FALSE(self->out_size_ < 1, NNACL_TENSOR_SIZE_INVALID);

  self->Release(self);

  TensorC *in0 = self->in_[0];
  NNACL_CHECK_FALSE(in0->shape_size_ < C1NUM, NNACL_GROUP_NORM_SHAPE_SIZE_INVALID);
  NNACL_CHECK_FALSE(in0->format_ != Format_NCHW, NNACL_GROUP_NORM_FORMAT_INVALID);

  param->unit_ = GetHeight(in0) * GetWidth(in0);
  param->batch_ = GetBatch(in0);
  param->channel_ = GetChannel(in0);
  return self->Prepare(self);
}

int GroupNormPrepare(struct KernelBase *self) {
  GroupNormStru *groupnorm = (GroupNormStru *)self;
  NNACL_CHECK_NULL_RETURN_ERR(groupnorm);
  GroupNormParameter *param = (GroupNormParameter *)groupnorm->base.param_;
  NNACL_CHECK_NULL_RETURN_ERR(param);
  NNACL_CHECK_FALSE(param->num_groups_ < 0, NNACL_GROUP_NORM_NUM_GROUPS_INVALID);
  NNACL_CHECK_FALSE(param->channel_ % param->num_groups_, NNACL_GROUP_NORM_NUM_GROUPS_INVALID);
  NNACL_CHECK_FALSE(param->num_groups_ == 0, NNACL_GROUP_NORM_NUM_GROUPS_INVALID);

  size_t mean_var_elem_num = param->num_groups_;
  param->mean_ = malloc(mean_var_elem_num * sizeof(float));
  param->variance_ = malloc(mean_var_elem_num * sizeof(float));
  if (param->mean_ == NULL || param->variance_ == NULL) {
    self->Release(self);
    return NNACL_MALLOC_BUFFER_FAILED;
  }
  return NNACL_OK;
}

int GroupNormRelease(struct KernelBase *self) {
  GroupNormStru *groupnorm = (GroupNormStru *)self;
  NNACL_CHECK_NULL_RETURN_ERR(groupnorm);
  GroupNormParameter *param = (GroupNormParameter *)groupnorm->base.param_;
  NNACL_CHECK_NULL_RETURN_ERR(param);

  if (param->mean_ != NULL) {
    free(param->mean_);
    param->mean_ = NULL;
  }
  if (param->variance_ != NULL) {
    free(param->variance_);
    param->variance_ = NULL;
  }

  return NNACL_OK;
}

int GroupNormImpl(void *param, int task_id, float l, float r) {
  NNACL_CHECK_NULL_RETURN_ERR(param);
  GroupNormStru *groupnorm_stru = (GroupNormStru *)param;
  GroupNormParameter *groupnorm_param = (GroupNormParameter *)groupnorm_stru->base.param_;
  NNACL_CHECK_NULL_RETURN_ERR(groupnorm_param);

  const void *input_data = groupnorm_stru->base.in_[0]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(input_data);
  const void *scale_data = groupnorm_stru->base.in_[C1NUM]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(scale_data);
  const void *offset_data = groupnorm_stru->base.in_[C2NUM]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(offset_data);
  void *output_data = groupnorm_stru->base.out_[0]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(output_data);

  NNACL_CHECK_NULL_RETURN_ERR(groupnorm_param->mean_);
  NNACL_CHECK_NULL_RETURN_ERR(groupnorm_param->variance_);

  int ret = GroupNormFp32(input_data, scale_data, offset_data, groupnorm_param->mean_, groupnorm_param->variance_,
                          groupnorm_param, task_id, output_data);

  return ret;
}

int GroupNormCompute(struct KernelBase *self) {
  return self->env_->ParallelLaunch(self->env_->thread_pool_, GroupNormImpl, self, self->param_->thread_num_);
}

KernelBase *CreateGroupNorm(OpParameter *param, int data_type) {
  GroupNormStru *groupnorm = (GroupNormStru *)malloc(sizeof(GroupNormStru));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(groupnorm);

  groupnorm->base.Prepare = GroupNormPrepare;
  groupnorm->base.Resize = GroupNormResize;
  groupnorm->base.Release = GroupNormRelease;
  groupnorm->base.Compute = GroupNormCompute;

  return (void *)groupnorm;
}

REG_KERNEL_CREATOR(PrimType_GroupNormFusion, kNumberTypeFloat32, CreateGroupNorm);
