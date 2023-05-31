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
#ifdef ENABLE_AVX
#include "nnacl/kernel/convolution_depthwise_sw_avx.h"
#include "nnacl/kernel/convolution_base.h"
#include "nnacl/fp32/conv_depthwise_fp32.h"
#include "nnacl/fp32/pack_fp32.h"

int ConvDwSWAVXInitPackedInputOutput(ConvolutionDepthwiseSWAVXStruct *conv_dw) {
  if (conv_dw->conv_.input_c_ % conv_dw->oc_tile_ != 0) {
    conv_dw->input_need_align_ = true;
    int ic_algin = UP_DIV(conv_dw->conv_.input_c_, conv_dw->oc_tile_);
    int input_hw = conv_dw->conv_.input_h_ * conv_dw->conv_.input_w_;
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv_dw->conv_.input_b_, input_hw, NNACL_ERR);
    int input_bhw = conv_dw->conv_.input_b_ * input_hw;
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(input_bhw, conv_dw->oc_tile_ * ic_algin, NNACL_ERR);
    int pack_input_size = input_bhw * conv_dw->oc_tile_ * ic_algin;
    conv_dw->packed_input_ =
      (float *)conv_dw->conv_.base_.env_->alloc(conv_dw->conv_.base_.env_->allocator_, pack_input_size * sizeof(float));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv_dw->packed_input_);
  }

  if (conv_dw->conv_.output_c_ % conv_dw->oc_tile_ != 0) {
    conv_dw->output_need_align_ = true;
    int oc_algin = UP_DIV(conv_dw->conv_.output_c_, conv_dw->oc_tile_);
    int output_hw = conv_dw->conv_.output_h_ * conv_dw->conv_.output_w_;
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv_dw->conv_.output_b_, output_hw, NNACL_ERR);
    int output_bhw = conv_dw->conv_.output_b_ * output_hw;
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(output_bhw, conv_dw->oc_tile_ * oc_algin, NNACL_ERR);
    int pack_output_size = output_bhw * conv_dw->oc_tile_ * oc_algin;
    conv_dw->packed_output_ = (float *)conv_dw->conv_.base_.env_->alloc(conv_dw->conv_.base_.env_->allocator_,
                                                                        pack_output_size * sizeof(float));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv_dw->packed_output_);
  }

  return NNACL_OK;
}

void ConvDwSWAVXPackWeight(ConvolutionBaseStruct *conv) {
  ConvolutionDepthwiseSWAVXStruct *conv_dw = (ConvolutionDepthwiseSWAVXStruct *)conv;
  NNACL_CHECK_NULL_RETURN_VOID(conv_dw);

  int oc_algin = UP_DIV(conv->output_c_, conv_dw->oc_tile_);
  void *origin_weight = conv->base_.train_session_ ? conv->base_.in_[SECOND_INPUT]->data_ : conv->origin_weight_;
  NNACL_CHECK_NULL_RETURN_VOID(origin_weight);

  PackNHWCToNXHWCXFp32(conv->kernel_h_, conv->kernel_w_, conv->output_c_, oc_algin, conv->input_c_,
                       (float *)conv->packed_weight_, (float *)conv->origin_weight_);
}

int ConvDwSWAVXMallocWeightBiasData(ConvolutionBaseStruct *conv) {
  ConvolutionDepthwiseSWAVXStruct *conv_dw = (ConvolutionDepthwiseSWAVXStruct *)conv;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw);

  int oc_algin = UP_DIV(conv->output_c_, conv_dw->oc_tile_);
  int pack_weight_size = oc_algin * conv_dw->oc_tile_ * conv->kernel_h_ * conv->kernel_w_;

  if (!conv->base_.train_session_) {
    NNACL_CHECK_MALLOC_SIZE(pack_weight_size * sizeof(float));
    conv->packed_weight_ = ConvBaseGetConvPackWeightData(conv, pack_weight_size * sizeof(float));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv->packed_weight_);
  }

  if (conv->base_.in_size_ == THREE_TENSOR) {
    int bias_size = oc_algin * conv_dw->oc_tile_;
    NNACL_CHECK_MALLOC_SIZE(bias_size * sizeof(float));
    conv->bias_data_ = conv->base_.env_->alloc(conv->base_.env_->allocator_, bias_size * sizeof(float));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv->bias_data_);
    memset(conv->bias_data_, 0, bias_size * sizeof(float));
  }
  return NNACL_OK;
}

int ConvDwSWAvxRun(void *cdata, int task_id, float l, float r) {
  ConvolutionDepthwiseSWAVXStruct *conv_dw = (ConvolutionDepthwiseSWAVXStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw);
  ConvParameter *conv_param = (ConvParameter *)conv_dw->conv_.base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);

  DepthwiseSWAvxFp32(conv_dw->packed_output_, conv_dw->packed_input_, (float *)conv_dw->conv_.packed_weight_,
                     (float *)conv_dw->conv_.bias_data_, conv_param, &conv_dw->sliding_param_, task_id);
  return NNACL_OK;
}

void ConvDwSWAVXFreePackedInputOutput(ConvolutionDepthwiseSWAVXStruct *conv_dw) {
  if (conv_dw->input_need_align_) {
    conv_dw->conv_.base_.env_->free(conv_dw->conv_.base_.env_->allocator_, conv_dw->packed_input_);
    conv_dw->packed_input_ = NULL;
  }
  if (conv_dw->output_need_align_) {
    conv_dw->conv_.base_.env_->free(conv_dw->conv_.base_.env_->allocator_, conv_dw->packed_output_);
    conv_dw->packed_output_ = NULL;
  }
}

int convolution_depthwise_sw_avx_compute(KernelBase *self) {
  ConvolutionDepthwiseSWAVXStruct *conv_dw = (ConvolutionDepthwiseSWAVXStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw);

  int ret = ConvDwSWAVXInitPackedInputOutput(conv_dw);
  if (ret != NNACL_OK) {
    ConvDwSWAVXFreePackedInputOutput(conv_dw);
    return ret;
  }

  ret = ConvBaseRepackWeight(&conv_dw->conv_);
  if (ret != NNACL_OK) {
    ConvDwSWAVXFreePackedInputOutput(conv_dw);
    return ret;
  }

  TensorC *input_tensor = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input_tensor);
  float *input_ptr = (float *)input_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(input_ptr);

  if (conv_dw->input_need_align_) {
    PackNHWCToNHWCXFp32(input_ptr, conv_dw->packed_input_, conv_dw->conv_.input_b_,
                        conv_dw->conv_.input_h_ * conv_dw->conv_.input_w_, conv_dw->conv_.input_c_, conv_dw->oc_tile_);
  } else {
    conv_dw->packed_input_ = input_ptr;
  }

  TensorC *output_tensor = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output_tensor);
  float *output_ptr = (float *)output_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(output_ptr);

  if (!conv_dw->output_need_align_) {
    conv_dw->packed_output_ = output_ptr;
  }

  ret = self->env_->parallel_launch(self->env_->thread_pool_, ConvDwSWAvxRun, self, self->thread_nr_);

  if (conv_dw->output_need_align_) {
    PackNHWCXToNHWCFp32(conv_dw->packed_output_, output_ptr, conv_dw->conv_.output_b_,
                        conv_dw->conv_.output_h_ * conv_dw->conv_.output_w_, conv_dw->conv_.output_c_,
                        conv_dw->oc_tile_);
  }

  ConvDwSWAVXFreePackedInputOutput(conv_dw);
  return ret;
}

int convolution_depthwise_sw_avx_prepare(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < TWO_TENSOR, NNACL_INPUT_TENSOR_ERROR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_OUTPUT_TENSOR_ERROR);

  ConvolutionDepthwiseSWAVXStruct *conv_dw = (ConvolutionDepthwiseSWAVXStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw);
  ConvParameter *conv_param = (ConvParameter *)self->param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);

  conv_dw->oc_tile_ = C8NUM;
  ConvBaseUpdateOriginWeightAndBias(&conv_dw->conv_);

  if (self->train_session_) {
    TensorC *weight_tensor = self->in_[SECOND_INPUT];
    NNACL_CHECK_NULL_RETURN_ERR(weight_tensor);
    NNACL_CHECK_TRUE_RET(weight_tensor->shape_size_ == DIMENSION_4D, NNACL_CONVOLUTION_WEIGHT_SHAPE_INVALID);

    int oc_algin = UP_DIV(conv_dw->conv_.output_c_, conv_dw->oc_tile_);
    int weight_size_hw = conv_dw->conv_.kernel_h_ * conv_dw->conv_.kernel_w_;
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(oc_algin * conv_dw->oc_tile_, weight_size_hw, NNACL_ERR);
    int pack_weight_size = oc_algin * conv_dw->oc_tile_ * weight_size_hw;
    self->work_size_ = pack_weight_size * sizeof(float);
  }

  return ConvBaseInitConvWeightBias(&conv_dw->conv_);
}

int convolution_depthwise_sw_avx_resize(KernelBase *self) {
  ConvolutionDepthwiseSWAVXStruct *conv_dw = (ConvolutionDepthwiseSWAVXStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw);
  ConvParameter *conv_param = (ConvParameter *)self->param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);

  ConvBasePrepare(&conv_dw->conv_);

  InitSlidingParamConvDw(&conv_dw->sliding_param_, conv_param, conv_dw->oc_tile_);
  return NNACL_OK;
}

int convolution_depthwise_sw_avx_release(KernelBase *self) {
  ConvolutionBaseStruct *conv = (ConvolutionBaseStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv);
  ConvBaseRelease(conv);
  return NNACL_OK;
}

KernelBase *CreateConvDwSWAVX(ConvParameter *conv_param) {
  ConvolutionDepthwiseSWAVXStruct *conv_dw =
    (ConvolutionDepthwiseSWAVXStruct *)malloc(sizeof(ConvolutionDepthwiseSWAVXStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(conv_dw);

  conv_dw->conv_.packed_weight_ = ConvDwSWAVXPackWeight;
  conv_dw->conv_.malloc_weight_bias_ = ConvDwSWAVXMallocWeightBiasData;

  conv_dw->conv_.base_.prepare = convolution_depthwise_sw_avx_prepare;
  conv_dw->conv_.base_.compute = convolution_depthwise_sw_avx_compute;
  conv_dw->conv_.base_.resize = convolution_depthwise_sw_avx_resize;
  conv_dw->conv_.base_.release = convolution_depthwise_sw_avx_release;
  return (KernelBase *)conv_dw;
}
#endif
