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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either convolutionress or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#if defined(ENABLE_ARM) || (defined(ENABLE_SSE) && !defined(ENABLE_AVX))
#include "nnacl/kernel/convolution_depthwise_3x3.h"
#include "nnacl/kernel/convolution_base.h"
#include "nnacl/fp32/conv_depthwise_fp32.h"
#include "nnacl/fp32/pack_fp32.h"

int ConvDw3x3Run(void *cdata, int task_id, float l, float r) {
  ConvolutionDepthwise3x3Struct *conv_dw = (ConvolutionDepthwise3x3Struct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw);

  int units = UP_DIV(conv_dw->conv_.compute_.out_w_, C2NUM);  // F(2, 3) contains 2 conv units
  int c4 = UP_ROUND(conv_dw->conv_.compute_.in_c_, C4NUM);
  int c12c4_units = C12NUM * c4 * units;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(c12c4_units, task_id, NNACL_ERR);
  float *buffer = conv_dw->buffer_ + c12c4_units * task_id;
  NNACL_CHECK_ZERO_RETURN_ERR(conv_dw->conv_.base_.thread_nr_);

  int step_oh = UP_DIV(conv_dw->conv_.compute_.out_h_, conv_dw->conv_.base_.thread_nr_);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(step_oh, task_id, NNACL_ERR);
  int start_oh = step_oh * task_id;
  int end_oh = MSMIN(start_oh + step_oh, conv_dw->conv_.compute_.out_h_);

  ConvParameter *conv_param = (ConvParameter *)conv_dw->conv_.base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);
  ConvDw3x3(conv_dw->output_ptr_, buffer, conv_dw->input_ptr_, (float *)conv_dw->conv_.packed_weight_,
            (float *)conv_dw->conv_.bias_data_, conv_param, start_oh, end_oh);
  return NNACL_OK;
}

void ConvDw3x3PackWeight(ConvolutionBaseStruct *conv) {
  void *origin_weight = (conv->base_.train_session_) ? conv->base_.in_[SECOND_INPUT]->data_ : conv->origin_weight_;
  NNACL_CHECK_NULL_RETURN_VOID(origin_weight);
  PackWeightConvDw3x3Fp32((float *)origin_weight, (float *)conv->packed_weight_, conv->compute_.out_c_);
}

int ConvDw3x3MallocWeightBiasData(ConvolutionBaseStruct *conv) {
  int c4 = UP_ROUND(conv->compute_.out_c_, C4NUM);
  if (!conv->base_.train_session_) {
    if (conv->packed_weight_ == NULL) {
      int pack_weight_size = c4 * C12NUM;
      NNACL_CHECK_MALLOC_SIZE(pack_weight_size * sizeof(float));
      conv->packed_weight_ = ConvBaseGetConvPackWeightData(conv, pack_weight_size * sizeof(float));
      NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv->packed_weight_);
    }
  }

  if (conv->bias_data_ == NULL) {
    NNACL_CHECK_MALLOC_SIZE(c4 * sizeof(float));
    conv->bias_data_ = conv->base_.env_->Alloc(conv->base_.env_->allocator_, c4 * sizeof(float));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv->bias_data_);
  }
  memset(conv->bias_data_, 0, c4 * sizeof(float));
  return NNACL_OK;
}

int ConvolutionDepthwise3x3Resize(KernelBase *self) {
  ConvolutionBaseStruct *conv = (ConvolutionBaseStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv);
  int ret = ConvBasePrepare(conv);
  if (ret != NNACL_OK) {
    return ret;
  }
  self->thread_nr_ = NNACL_MIN(self->thread_nr_, conv->compute_.out_h_);
  return NNACL_OK;
}

int ConvolutionDepthwise3x3Prepare(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < TWO_TENSOR, NNACL_INPUT_TENSOR_ERROR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_OUTPUT_TENSOR_ERROR);

  ConvolutionDepthwise3x3Struct *conv_dw = (ConvolutionDepthwise3x3Struct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw);

  ConvBaseUpdateOriginWeightAndBias(&conv_dw->conv_);

  if (self->train_session_) {
    int c4 = UP_ROUND(conv_dw->conv_.compute_.out_c_, C4NUM);
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(c4, C12NUM, NNACL_ERR);
    int pack_weight_size = c4 * C12NUM;
    self->work_size_ = pack_weight_size * sizeof(float);
  }

  return ConvBaseInitConvWeightBias(&conv_dw->conv_);
}

int ConvolutionDepthwise3x3Compute(KernelBase *self) {
  ConvolutionDepthwise3x3Struct *conv_dw = (ConvolutionDepthwise3x3Struct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw);

  int units = UP_DIV(conv_dw->conv_.compute_.out_w_, C2NUM);  // F(2, 3) contains 2 conv units
  int c4 = UP_ROUND(conv_dw->conv_.compute_.in_c_, C4NUM);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(C12NUM, c4, NNACL_ERR);
  int c12c4 = C12NUM * c4;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(c12c4, units, NNACL_ERR);
  int c12c4_units = c12c4 * units;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(c12c4_units, self->thread_nr_, NNACL_ERR);
  int buffer_size = c12c4_units * self->thread_nr_;

  conv_dw->buffer_ = self->env_->Alloc(self->env_->allocator_, buffer_size * sizeof(float));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv_dw->buffer_);

  int ret = ConvBaseRepackWeight(&conv_dw->conv_);
  if (ret != NNACL_OK) {
    self->env_->Free(self->env_->allocator_, conv_dw->buffer_);
    return ret;
  }

  conv_dw->input_ptr_ = self->in_[FIRST_INPUT]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw->input_ptr_);
  conv_dw->output_ptr_ = self->out_[OUTPUT_INDEX]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw->output_ptr_);

  ret = self->env_->ParallelLaunch(self->env_->thread_pool_, ConvDw3x3Run, self, self->thread_nr_);
  self->env_->Free(self->env_->allocator_, conv_dw->buffer_);
  return ret;
}

int ConvolutionDepthwise3x3Release(KernelBase *self) {
  ConvolutionBaseStruct *conv = (ConvolutionBaseStruct *)self;
  ConvBaseRelease(conv);
  return NNACL_OK;
}

KernelBase *CreateConvDw3x3(ConvParameter *conv_param) {
  ConvolutionDepthwise3x3Struct *conv_dw =
    (ConvolutionDepthwise3x3Struct *)malloc(sizeof(ConvolutionDepthwise3x3Struct));
  NNACL_CHECK_NULL_RETURN_NULL(conv_dw);
  memset(conv_dw, 0, sizeof(ConvolutionDepthwise3x3Struct));
  conv_dw->conv_.pack_weight_ = ConvDw3x3PackWeight;
  conv_dw->conv_.malloc_weight_bias_ = ConvDw3x3MallocWeightBiasData;
  conv_dw->conv_.base_.Resize = ConvolutionDepthwise3x3Resize;
  conv_dw->conv_.base_.Prepare = ConvolutionDepthwise3x3Prepare;
  conv_dw->conv_.base_.Compute = ConvolutionDepthwise3x3Compute;
  conv_dw->conv_.base_.Release = ConvolutionDepthwise3x3Release;

  return (KernelBase *)conv_dw;
}
#endif
