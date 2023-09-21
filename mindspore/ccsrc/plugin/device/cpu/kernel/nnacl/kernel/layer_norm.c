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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either log_softmaxress or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nnacl/kernel/layer_norm.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/tensor_c_utils.h"
#include "nnacl/fp32/layer_norm_fp32.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/layer_norm_fp16.h"
#endif

int LayerNormRun(void *cdata, int task_id, float l, float r) {
  LayerNormStruct *ln = (LayerNormStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(ln);
  if (ln->data_type_ == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
    return LayerNormFp16(ln->src_data_, ln->gamma_data_, ln->beta_data_, ln->dst_data_, ln->mean_data_, ln->var_data_,
                         &ln->compute_, task_id, ln->base_.thread_nr_);
#endif
  }
  return LayerNorm(ln->src_data_, ln->gamma_data_, ln->beta_data_, ln->dst_data_, ln->mean_data_, ln->var_data_,
                   &ln->compute_, task_id, ln->base_.thread_nr_);
}

int LayerNormResize(KernelBase *self) {
  LayerNormStruct *layer_norm = (LayerNormStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(layer_norm);
  LayerNormComputeParam *compute = &layer_norm->compute_;

  TensorC *input = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input);

  if (compute->begin_norm_axis_ < 0) {
    compute->begin_norm_axis_ = compute->begin_norm_axis_ + (int)input->shape_size_;
  }

  if (compute->begin_params_axis_ < 0) {
    compute->begin_params_axis_ = compute->begin_params_axis_ + (int)input->shape_size_;
  }

  compute->norm_outer_size_ = 1;
  for (int i = 0; i < compute->begin_norm_axis_; ++i) {
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(compute->norm_outer_size_, input->shape_[i], NNACL_ERR);
    compute->norm_outer_size_ *= input->shape_[i];
  }

  compute->norm_inner_size_ = 1;
  for (size_t i = compute->begin_norm_axis_; i < input->shape_size_; ++i) {
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(compute->norm_inner_size_, input->shape_[i], NNACL_ERR);
    compute->norm_inner_size_ *= input->shape_[i];
  }

  compute->params_outer_size_ = 1;
  for (int i = 0; i < compute->begin_params_axis_; ++i) {
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(compute->params_outer_size_, input->shape_[i], NNACL_ERR);
    compute->params_outer_size_ *= input->shape_[i];
  }

  compute->params_inner_size_ = 1;
  for (size_t i = compute->begin_params_axis_; i < input->shape_size_; ++i) {
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(compute->params_inner_size_, input->shape_[i], NNACL_ERR);
    compute->params_inner_size_ *= input->shape_[i];
  }

  int out_num = GetElementNum(self->out_[OUTPUT_INDEX]);
  self->thread_nr_ = self->UpdateThread(TC_PTYPE(PrimType_LayerNormFusion), compute->norm_inner_size_,
                                        compute->norm_inner_size_, out_num, self->thread_nr_);
  self->thread_nr_ = NNACL_MIN(compute->norm_outer_size_, self->thread_nr_);
  return NNACL_OK;
}

int LayerNormCompute(KernelBase *self) {
  LayerNormStruct *layer_norm = (LayerNormStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(layer_norm);

  layer_norm->src_data_ = self->in_[FIRST_INPUT]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(layer_norm->src_data_);
  layer_norm->gamma_data_ = self->in_[SECOND_INPUT]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(layer_norm->gamma_data_);
  layer_norm->beta_data_ = self->in_[THIRD_INPUT]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(layer_norm->beta_data_);
  layer_norm->dst_data_ = self->out_[OUTPUT_INDEX]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(layer_norm->dst_data_);

  if (layer_norm->base_.out_size_ == THREE_TENSOR) {
    layer_norm->mean_data_ = self->out_[Index1]->data_;
    NNACL_CHECK_NULL_RETURN_ERR(layer_norm->mean_data_);
    layer_norm->var_data_ = self->out_[Index2]->data_;
    NNACL_CHECK_NULL_RETURN_ERR(layer_norm->var_data_);
  } else if (layer_norm->base_.out_size_ != ONE_TENSOR) {
    return NNACL_LAYER_NORM_OUTPUT_NUM_INVALID;
  }

  return self->env_->ParallelLaunch(self->env_->thread_pool_, LayerNormRun, self, self->thread_nr_);
}

KernelBase *CreateLayerNorm(OpParameter *param, int data_type) {
  LayerNormStruct *layer_norm = (LayerNormStruct *)malloc(sizeof(LayerNormStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(layer_norm);
  memset(layer_norm, 0, sizeof(LayerNormStruct));
  layer_norm->data_type_ = data_type;

  LayerNormParameter *layer_norm_param = (LayerNormParameter *)param;
  layer_norm->compute_.epsilon_ = layer_norm_param->epsilon_;
  layer_norm->compute_.elementwise_affine_ = layer_norm_param->elementwise_affine_;
  layer_norm->compute_.begin_norm_axis_ = layer_norm_param->begin_norm_axis_;
  layer_norm->compute_.begin_params_axis_ = layer_norm_param->begin_params_axis_;

  layer_norm->base_.Prepare = DefaultPrepare3In1Out;
  layer_norm->base_.Release = DefaultRelease;
  layer_norm->base_.Resize = LayerNormResize;
  layer_norm->base_.Compute = LayerNormCompute;
  return (KernelBase *)layer_norm;
}

REG_KERNEL_CREATOR(PrimType_LayerNormFusion, kNumberTypeFloat16, CreateLayerNorm)
REG_KERNEL_CREATOR(PrimType_LayerNormFusion, kNumberTypeFloat32, CreateLayerNorm)
