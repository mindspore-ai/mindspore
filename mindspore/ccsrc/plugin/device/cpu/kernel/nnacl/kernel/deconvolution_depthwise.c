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

#include "nnacl/kernel/deconvolution_depthwise.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/fp32/conv_depthwise_fp32.h"
#include "nnacl/fp32/pack_fp32.h"

int DeConvDwInitPackedInputOutput(DeConvDwStruct *deconv_dw) {
  if (!deconv_dw->need_align_) {
    return NNACL_OK;
  }
  ExecEnv *env = deconv_dw->conv_.base_.env_;
  ConvComputeParam *compute = &deconv_dw->conv_.compute_;

  int ic4 = UP_ROUND(compute->in_c_, compute->tile_num_);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(compute->in_n_, compute->in_hw_, NNACL_ERR);
  int input_bhw = compute->in_n_ * compute->in_hw_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(input_bhw, ic4, NNACL_ERR);
  int pack_input_size = input_bhw * ic4;
  deconv_dw->packed_input_ = (float *)env->alloc(env->allocator_, pack_input_size * sizeof(float));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(deconv_dw->packed_input_);

  int oc4 = UP_ROUND(compute->out_c_, compute->tile_num_);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(compute->out_n_, compute->out_hw_, NNACL_ERR);
  int output_bhw = compute->out_n_ * compute->out_hw_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(output_bhw, oc4, NNACL_ERR);
  int pack_output_size = output_bhw * oc4;
  deconv_dw->packed_output_ = (float *)env->alloc(env->allocator_, pack_output_size * sizeof(float));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(deconv_dw->packed_output_);
  memset(deconv_dw->packed_output_, 0, pack_output_size * sizeof(float));

  return NNACL_OK;
}

int DeconvDwRun(void *cdata, int task_id, float l, float r) {
  DeConvDwStruct *deconv_dw = (DeConvDwStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(deconv_dw);
  DeconvDwSWFp32(deconv_dw->packed_output_, deconv_dw->packed_input_, (float *)deconv_dw->conv_.packed_weight_,
                 (float *)deconv_dw->conv_.bias_data_, (ConvParameter *)deconv_dw->conv_.base_.param_,
                 &deconv_dw->sliding_, task_id);
  return NNACL_OK;
}

int DeConvDwMallocWeightBiasData(ConvolutionBaseStruct *conv) {
  DeConvDwStruct *deconv_dw = (DeConvDwStruct *)conv;
  NNACL_CHECK_NULL_RETURN_ERR(deconv_dw);

  int oc4 = UP_ROUND(conv->compute_.out_c_, conv->compute_.tile_num_);
  if (!conv->base_.train_session_) {
    int pack_weight_size = oc4 * conv->compute_.kernel_hw_;
    NNACL_CHECK_MALLOC_SIZE(pack_weight_size);
    deconv_dw->conv_.packed_weight_ = ConvBaseGetConvPackWeightData(conv, pack_weight_size * sizeof(float));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(deconv_dw->conv_.packed_weight_);
  }

  if (deconv_dw->conv_.bias_data_ == NULL) {
    NNACL_CHECK_MALLOC_SIZE(oc4 * sizeof(float));
    deconv_dw->conv_.bias_data_ = conv->base_.env_->alloc(conv->base_.env_->allocator_, oc4 * sizeof(float));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(deconv_dw->conv_.bias_data_);
  }
  memset(deconv_dw->conv_.bias_data_, 0, oc4 * sizeof(float));
  return NNACL_OK;
}

void DeConvDwPackWeight(ConvolutionBaseStruct *conv) {
  void *origin_weight = conv->base_.train_session_ ? conv->base_.in_[SECOND_INPUT]->data_ : conv->origin_weight_;
  NNACL_CHECK_NULL_RETURN_VOID(origin_weight);
  PackNCHWToNC4HW4Fp32(origin_weight, conv->pack_weight_, 1, conv->compute_.kernel_hw_, conv->compute_.out_c_);
}

void DeConvDwFreePackedInputOutput(DeConvDwStruct *deconv_dw) {
  if (deconv_dw->need_align_) {
    ExecEnv *env = deconv_dw->conv_.base_.env_;

    env->free(env->allocator_, deconv_dw->packed_input_);
    deconv_dw->packed_input_ = NULL;
    env->free(env->allocator_, deconv_dw->packed_output_);
    deconv_dw->packed_output_ = NULL;
  }
}

int deconv_dw_prepare(KernelBase *self) {
  DeConvDwStruct *deconv_dw = (DeConvDwStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(deconv_dw);
  ConvComputeParam *compute = &deconv_dw->conv_.compute_;
  deconv_dw->conv_.compute_.tile_num_ = C4NUM;

  NNACL_CHECK_FALSE(self->in_size_ < TWO_TENSOR, NNACL_ERR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_ERR);

  NNACL_CHECK_FALSE(compute->in_c_ != compute->out_c_, NNACL_DECONVOLUTION_DEPTHWISE_CHANNEL_INVALID);
  NNACL_CHECK_FALSE(compute->dilation_h_ != Num1, NNACL_DECONVOLUTION_DEPTHWISE_DILATION_INVALID);
  NNACL_CHECK_FALSE(compute->dilation_w_ != Num1, NNACL_DECONVOLUTION_DEPTHWISE_DILATION_INVALID);

  ConvBaseUpdateOriginWeightAndBias(&deconv_dw->conv_);

  if (self->train_session_) {
    int oc4 = UP_ROUND(compute->out_c_, compute->tile_num_);
    int pack_weight_size = oc4 * compute->kernel_hw_;
    self->work_size_ = pack_weight_size;
  }

  int ret = ConvBaseInitConvWeightBias(&deconv_dw->conv_);
  if (ret != NNACL_OK) {
    return ret;
  }

  NNACL_CHECK_NULL_RETURN_ERR(deconv_dw->conv_.packed_weight_);
  NNACL_CHECK_NULL_RETURN_ERR(deconv_dw->conv_.bias_data_);
  return NNACL_OK;
}

int deconv_dw_resize(KernelBase *self) {
  DeConvDwStruct *deconv_dw = (DeConvDwStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(deconv_dw);

  (void)ConvBaseUpdateComputeInfo(&deconv_dw->conv_);

  int ret = ConvBaseCheckResizeValid(&deconv_dw->conv_);
  if (ret != NNACL_OK) {
    return ret;
  }

  ret = ConvBasePrepare(&deconv_dw->conv_);
  if (ret != NNACL_OK) {
    return ret;
  }

  int tile_num = deconv_dw->conv_.compute_.tile_num_;
  (void)InitSlidingParamConvDw(&deconv_dw->sliding_, (ConvParameter *)self->param_, tile_num);
  self->thread_nr_ = NNACL_MIN(self->thread_nr_, UP_DIV(deconv_dw->conv_.compute_.out_c_, tile_num));
  deconv_dw->need_align_ = deconv_dw->conv_.compute_.in_c_ % tile_num != 0;

  return NNACL_OK;
}

int deconv_dw_compute(KernelBase *self) {
  DeConvDwStruct *deconv_dw = (DeConvDwStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(deconv_dw);
  ConvComputeParam *compute = &deconv_dw->conv_.compute_;

  TensorC *in_tensor = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(in_tensor);
  float *in_data = (float *)in_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(in_data);
  TensorC *out_tensor = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(out_tensor);
  float *out_data = (float *)out_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(out_data);

  int ret = ConvBaseRepackWeight(&deconv_dw->conv_);
  if (ret != NNACL_OK) {
    return ret;
  }

  ret = DeConvDwInitPackedInputOutput(deconv_dw);
  if (ret != NNACL_OK) {
    DeConvDwFreePackedInputOutput(deconv_dw);
    return ret;
  }

  if (deconv_dw->need_align_) {
    PackNHWCToNHWC4Fp32(in_data, deconv_dw->packed_input_, compute->in_n_, compute->in_hw_, compute->in_c_);
  } else {
    deconv_dw->packed_input_ = in_data;
    deconv_dw->packed_output_ = out_data;
    memset(deconv_dw->packed_output_, 0, GetSize(out_tensor));
  }

  ret = self->env_->parallel_launch(self->env_->thread_pool_, DeconvDwRun, self, self->thread_nr_);

  if (deconv_dw->need_align_) {
    PackNHWCXToNHWCFp32(deconv_dw->packed_output_, out_data, compute->out_n_, compute->out_hw_, compute->out_c_,
                        compute->tile_num_);
  }
  DeConvDwFreePackedInputOutput(deconv_dw);
  return ret;
}

ConvolutionBaseStruct *CreateDeConvDw(ConvParameter *param) {
  DeConvDwStruct *deconv_dw = (DeConvDwStruct *)malloc(sizeof(DeConvDwStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(deconv_dw);
  memset(deconv_dw, 0, sizeof(DeConvDwStruct));

  deconv_dw->conv_.pack_weight_ = DeConvDwPackWeight;
  deconv_dw->conv_.malloc_weight_bias_ = DeConvDwMallocWeightBiasData;
  deconv_dw->conv_.base_.prepare = deconv_dw_prepare;
  deconv_dw->conv_.base_.resize = deconv_dw_resize;
  deconv_dw->conv_.base_.release = default_release;
  deconv_dw->conv_.base_.compute = deconv_dw_compute;
  return &deconv_dw->conv_;
}
