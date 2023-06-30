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

#include "nnacl/kernel/convolution_depthwise_sw.h"
#include "nnacl/kernel/convolution_base.h"
#include "nnacl/fp32/conv_depthwise_fp32.h"
#include "nnacl/fp32/pack_fp32.h"

int ConvDwSWMallocWeightBiasData(ConvolutionBaseStruct *conv) {
  int OC4 = UP_DIV(conv->compute_.out_c_, C4NUM);
  int pack_weight_size = C4NUM * OC4 * conv->compute_.kernel_hw_;
  if (!conv->base_.train_session_) {
    NNACL_CHECK_MALLOC_SIZE(pack_weight_size * sizeof(float));
    conv->packed_weight_ = ConvBaseGetConvPackWeightData(conv, pack_weight_size * sizeof(float));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv->packed_weight_);
  }

  int malloc_size = NNACL_MAX(conv->compute_.out_c_, C4NUM * OC4);
  if (conv->bias_data_ == NULL) {
    NNACL_CHECK_MALLOC_SIZE(malloc_size * sizeof(float));
    conv->bias_data_ = conv->base_.env_->Alloc(conv->base_.env_->allocator_, malloc_size * sizeof(float));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv->bias_data_);
  }
  memset(conv->bias_data_, 0, malloc_size * sizeof(float));
  conv->base_.thread_nr_ = NNACL_MIN(conv->base_.thread_nr_, OC4);
  return NNACL_OK;
}

int ConvDwSWInitPackedInputOutput(ConvolutionDepthwiseSWStruct *conv_dw) {
  if (conv_dw->conv_.compute_.in_c_ % C4NUM == 0) {
    conv_dw->need_align_ = false;
    return NNACL_OK;
  }

  conv_dw->need_align_ = true;
  int IC4 = UP_DIV(conv_dw->conv_.compute_.in_c_, C4NUM);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv_dw->conv_.compute_.in_n_, conv_dw->conv_.compute_.in_hw_, NNACL_ERR);
  int conv_input_bhw = conv_dw->conv_.compute_.in_n_ * conv_dw->conv_.compute_.in_hw_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv_input_bhw, C4NUM * IC4, NNACL_ERR);
  int pack_input_size = conv_input_bhw * C4NUM * IC4;
  NNACL_CHECK_MALLOC_SIZE(pack_input_size * sizeof(float));
  conv_dw->packed_input_ =
    (float *)conv_dw->conv_.base_.env_->Alloc(conv_dw->conv_.base_.env_->allocator_, pack_input_size * sizeof(float));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv_dw->packed_input_);

  int OC4 = UP_DIV(conv_dw->conv_.compute_.out_c_, C4NUM);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv_dw->conv_.compute_.out_n_, conv_dw->conv_.compute_.out_hw_, NNACL_ERR);
  int output_bhw = conv_dw->conv_.compute_.out_n_ * conv_dw->conv_.compute_.out_hw_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(output_bhw, C4NUM * OC4, NNACL_ERR);
  int pack_output_size = output_bhw * C4NUM * OC4;
  NNACL_CHECK_MALLOC_SIZE(pack_output_size * sizeof(float));
  conv_dw->packed_output_ =
    (float *)conv_dw->conv_.base_.env_->Alloc(conv_dw->conv_.base_.env_->allocator_, pack_output_size * sizeof(float));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv_dw->packed_output_);
  return NNACL_OK;
}

int ConvDwSWRun(void *cdata, int task_id, float l, float r) {
  ConvolutionDepthwiseSWStruct *conv_dw = (ConvolutionDepthwiseSWStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw);
  ConvParameter *conv_param = (ConvParameter *)conv_dw->conv_.base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);

  ConvDwSWFp32(conv_dw->packed_output_, conv_dw->packed_input_, (float *)conv_dw->conv_.packed_weight_,
               (float *)conv_dw->conv_.bias_data_, conv_param, &conv_dw->sliding_, task_id);
  return NNACL_OK;
}

void ConvDwSWFreePackedInputOutput(ConvolutionDepthwiseSWStruct *conv_dw) {
  if (conv_dw->need_align_) {
    conv_dw->conv_.base_.env_->Free(conv_dw->conv_.base_.env_->allocator_, conv_dw->packed_input_);
    conv_dw->packed_input_ = NULL;
    conv_dw->conv_.base_.env_->Free(conv_dw->conv_.base_.env_->allocator_, conv_dw->packed_output_);
    conv_dw->packed_output_ = NULL;
  }
}

void ConvDwSWPackWeight(ConvolutionBaseStruct *conv) {
  void *origin_weight = (conv->base_.train_session_) ? conv->base_.in_[SECOND_INPUT]->data_ : conv->origin_weight_;
  NNACL_CHECK_NULL_RETURN_VOID(origin_weight);
  PackNCHWToNC4HW4Fp32(origin_weight, conv->packed_weight_, 1, conv->compute_.kernel_hw_, conv->compute_.out_c_);
}

int ConvolutionDepthwiseSWResize(KernelBase *self) {
  ConvolutionDepthwiseSWStruct *conv_dw = (ConvolutionDepthwiseSWStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw);
  int ret = ConvBasePrepare(&conv_dw->conv_);
  if (ret != NNACL_OK) {
    return ret;
  }

  ConvParameter *conv_param = (ConvParameter *)self->param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);
  InitSlidingParamConvDw(&conv_dw->sliding_, conv_param, C4NUM);

  self->thread_nr_ = NNACL_MIN(self->thread_nr_, conv_dw->conv_.compute_.out_h_);
  NNACL_CHECK_ZERO_RETURN_ERR(self->thread_nr_);
  return NNACL_OK;
}

int ConvolutionDepthwiseSWCompute(KernelBase *self) {
  ConvolutionDepthwiseSWStruct *conv_dw = (ConvolutionDepthwiseSWStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw);

  int ret = ConvDwSWInitPackedInputOutput(conv_dw);
  if (ret != NNACL_OK) {
    ConvDwSWFreePackedInputOutput(conv_dw);
    return ret;
  }

  ret = ConvBaseRepackWeight(&conv_dw->conv_);
  if (ret != NNACL_OK) {
    ConvDwSWFreePackedInputOutput(conv_dw);
    return ret;
  }

  TensorC *input_tensor = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input_tensor);
  float *input_ptr = (float *)input_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(input_ptr);
  if (conv_dw->need_align_) {
    PackNHWCToNHWC4Fp32(input_ptr, conv_dw->packed_input_, conv_dw->conv_.compute_.in_n_,
                        conv_dw->conv_.compute_.in_hw_, conv_dw->conv_.compute_.in_c_);
  } else {
    conv_dw->packed_input_ = input_ptr;
  }

  TensorC *output_tensor = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output_tensor);
  float *output_ptr = (float *)output_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(output_ptr);
  if (!conv_dw->need_align_) {
    conv_dw->packed_output_ = output_ptr;
  }

  ret = self->env_->ParallelLaunch(self->env_->thread_pool_, ConvDwSWRun, self, self->thread_nr_);

  if (conv_dw->need_align_) {
    PackNHWCXToNHWCFp32(conv_dw->packed_output_, output_ptr, conv_dw->conv_.compute_.out_n_,
                        conv_dw->conv_.compute_.out_hw_, conv_dw->conv_.compute_.out_c_, C4NUM);
  }

  ConvDwSWFreePackedInputOutput(conv_dw);
  return ret;
}

int ConvolutionDdepthwiseSWPrepare(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < TWO_TENSOR, NNACL_INPUT_TENSOR_ERROR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_OUTPUT_TENSOR_ERROR);

  ConvolutionDepthwiseSWStruct *conv_dw = (ConvolutionDepthwiseSWStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw);
  ConvParameter *conv_param = (ConvParameter *)self->param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);

  ConvBaseUpdateOriginWeightAndBias(&conv_dw->conv_);

  if (self->train_session_) {
    int OC4 = UP_DIV(conv_dw->conv_.compute_.out_c_, C4NUM);
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(C4NUM * OC4, conv_dw->conv_.compute_.kernel_hw_, NNACL_ERR);
    int pack_weight_size = C4NUM * OC4 * conv_dw->conv_.compute_.kernel_hw_;
    self->work_size_ = pack_weight_size * sizeof(float);
  }

  return ConvBaseInitConvWeightBias(&conv_dw->conv_);
}

int ConvolutionDepthwiseSWRelease(KernelBase *self) {
  ConvolutionBaseStruct *conv = (ConvolutionBaseStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv);
  ConvBaseRelease(conv);
  return NNACL_OK;
}

KernelBase *CreateConvDwSW(ConvParameter *conv_param) {
  ConvolutionDepthwiseSWStruct *conv_dw = (ConvolutionDepthwiseSWStruct *)malloc(sizeof(ConvolutionDepthwiseSWStruct));
  NNACL_CHECK_NULL_RETURN_NULL(conv_dw);
  memset(conv_dw, 0, sizeof(ConvolutionDepthwiseSWStruct));

  conv_dw->conv_.malloc_weight_bias_ = ConvDwSWMallocWeightBiasData;
  conv_dw->conv_.pack_weight_ = ConvDwSWPackWeight;
  conv_dw->conv_.base_.Resize = ConvolutionDepthwiseSWResize;
  conv_dw->conv_.base_.Compute = ConvolutionDepthwiseSWCompute;
  conv_dw->conv_.base_.Prepare = ConvolutionDdepthwiseSWPrepare;
  conv_dw->conv_.base_.Release = ConvolutionDepthwiseSWRelease;
  return (KernelBase *)conv_dw;
}
