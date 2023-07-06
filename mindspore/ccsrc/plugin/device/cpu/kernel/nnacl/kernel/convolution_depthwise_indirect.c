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

#include "nnacl/kernel/convolution_depthwise_indirect.h"
#include "nnacl/kernel/convolution_base.h"
#include "nnacl/fp32/conv_depthwise_fp32.h"
#include "nnacl/fp32/pack_fp32.h"

int ConvDwIndirectMallocIndirectBuffer(ConvolutionDepthwiseIndirectStruct *conv_dw) {
  ConvComputeParam *compute = &conv_dw->conv_.compute_;
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(compute);

  conv_dw->step_w_ = compute->dilation_w_ == 1 ? compute->stride_w_ : compute->kernel_w_;
  int step_w_2d = conv_dw->step_w_ * compute->kernel_h_;
  conv_dw->step_h_ = (compute->kernel_h_ * compute->kernel_w_) + (compute->out_w_ - 1) * step_w_2d;
  int step_h_2d = compute->out_h_ * conv_dw->step_h_;
  int buffer_size = compute->out_n_ * step_h_2d;

  ExecEnv *env = conv_dw->conv_.base_.env_;
  conv_dw->indirect_buffer_ = (float **)(env->Alloc(env->allocator_, buffer_size * sizeof(float *)));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv_dw->indirect_buffer_);
  return NNACL_OK;
}

int ConvDwIndirectRun(void *cdata, int task_id, float l, float r) {
  ConvolutionDepthwiseIndirectStruct *conv_dw = (ConvolutionDepthwiseIndirectStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw);
  ConvParameter *conv_param = (ConvParameter *)conv_dw->conv_.base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);

  ConvDwIndirection(conv_dw->output_ptr_, conv_dw->indirect_buffer_, (float *)conv_dw->conv_.packed_weight_,
                    (float *)conv_dw->conv_.bias_data_, conv_dw->zero_ptr_, conv_param, task_id);
  return NNACL_OK;
}

int ConvDwIndirectMallocPackedInput(ConvolutionDepthwiseIndirectStruct *conv_dw) {
  int IC_DIV = UP_DIV(conv_dw->conv_.compute_.in_c_, conv_dw->div_flag_);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv_dw->conv_.compute_.in_n_, conv_dw->conv_.compute_.in_hw_, NNACL_ERR);
  int conv_input_bhw = conv_dw->conv_.compute_.in_n_ * conv_dw->conv_.compute_.in_hw_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv_input_bhw, conv_dw->div_flag_ * IC_DIV, NNACL_ERR);
  int pack_input_size = conv_input_bhw * conv_dw->div_flag_ * IC_DIV;
  conv_dw->packed_input_ =
    conv_dw->conv_.base_.env_->Alloc(conv_dw->conv_.base_.env_->allocator_, pack_input_size * sizeof(float));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv_dw->packed_input_);
  return NNACL_OK;
}

void ConvDwIndirectPackWeight(ConvolutionBaseStruct *conv) {
  TensorC *weight_tensor = conv->base_.in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_VOID(weight_tensor);
  void *origin_weight = (conv->base_.train_session_) ? weight_tensor->data_ : conv->origin_weight_;
  NNACL_CHECK_NULL_RETURN_VOID(origin_weight);

#ifdef ENABLE_AVX
  PackDepthwiseIndirectWeightC8Fp32(origin_weight, conv->packed_weight_, conv->compute_.kernel_h_,
                                    conv->compute_.kernel_w_, conv->compute_.out_c_);
#else
  PackDepthwiseIndirectWeightC4Fp32(origin_weight, conv->packed_weight_, conv->compute_.kernel_h_,
                                    conv->compute_.kernel_w_, conv->compute_.out_c_);
#endif
}

int ConvDwIndirectMallocWeightBiasData(ConvolutionBaseStruct *conv) {
  ConvolutionDepthwiseIndirectStruct *conv_dw = (ConvolutionDepthwiseIndirectStruct *)conv;
  ExecEnv *env = conv->base_.env_;

  int batch_flag = UP_DIV(conv->compute_.out_c_, conv_dw->div_flag_);
  int pack_weight_size = conv_dw->div_flag_ * batch_flag * conv->compute_.kernel_hw_;
  if (!conv->base_.train_session_) {
    NNACL_CHECK_MALLOC_SIZE(pack_weight_size * sizeof(float));
    conv->packed_weight_ = ConvBaseGetConvPackWeightData(conv, pack_weight_size * sizeof(float));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv->packed_weight_);
  }

  // malloc zero ptr
  NNACL_CHECK_MALLOC_SIZE(batch_flag * conv_dw->div_flag_ * sizeof(float));
  conv_dw->zero_ptr_ = (float *)env->Alloc(env->allocator_, batch_flag * conv_dw->div_flag_ * sizeof(float));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv_dw->zero_ptr_);
  memset(conv_dw->zero_ptr_, 0, batch_flag * conv_dw->div_flag_ * sizeof(float));

  // malloc bias ptr
  if (conv->bias_data_ == NULL) {
    conv->bias_data_ = env->Alloc(env->allocator_, batch_flag * conv_dw->div_flag_ * sizeof(float));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv->bias_data_);
  }
  memset(conv->bias_data_, 0, batch_flag * conv_dw->div_flag_ * sizeof(float));
  return NNACL_OK;
}

int ConvolutionDepthwiseIndirectCompute(KernelBase *self) {
  ConvolutionDepthwiseIndirectStruct *conv_dw = (ConvolutionDepthwiseIndirectStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw);
  TensorC *input_tensor = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input_tensor);
  void *input_ptr = input_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(input_ptr);

  if (conv_dw->conv_.compute_.in_c_ % conv_dw->div_flag_ != 0) {
    int ret = ConvDwIndirectMallocPackedInput(conv_dw);
    if (ret != NNACL_OK) {
      return ret;
    }
#ifdef ENABLE_AVX
    PackNHWCToNHWC8Fp32(input_ptr, conv_dw->packed_input_, conv_dw->conv_.compute_.in_n_,
                        conv_dw->conv_.compute_.in_hw_, conv_dw->conv_.compute_.in_c_);
#else
    PackNHWCToNHWC4Fp32(input_ptr, conv_dw->packed_input_, conv_dw->conv_.compute_.in_n_,
                        conv_dw->conv_.compute_.in_hw_, conv_dw->conv_.compute_.in_c_);
#endif
  } else {
    conv_dw->packed_input_ = input_ptr;
  }

  int ret = ConvBaseRepackWeight(&conv_dw->conv_);
  if (ret != NNACL_OK) {
    return ret;
  }

  TensorC *output_tensor = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output_tensor);
  conv_dw->output_ptr_ = output_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw->output_ptr_);
  ConvParameter *conv_param = (ConvParameter *)self->param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);

  ConvDwInitIndirection(conv_dw->indirect_buffer_, conv_dw->packed_input_, conv_dw->zero_ptr_, conv_param,
                        conv_dw->step_h_, conv_dw->step_w_);

  ret = self->env_->ParallelLaunch(self->env_->thread_pool_, ConvDwIndirectRun, self, self->thread_nr_);

  if (conv_dw->conv_.compute_.in_c_ % conv_dw->div_flag_ != 0) {
    self->env_->Free(self->env_->allocator_, conv_dw->packed_input_);
  }
  return ret;
}
int ConvolutionDepthwiseIndirectResize(KernelBase *self) {
  ConvolutionDepthwiseIndirectStruct *conv_dw = (ConvolutionDepthwiseIndirectStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw);

  if (conv_dw->indirect_buffer_ != NULL) {
    self->env_->Free(self->env_->allocator_, conv_dw->indirect_buffer_);
    conv_dw->indirect_buffer_ = NULL;
  }

  int ret = ConvBasePrepare(&conv_dw->conv_);
  if (ret != NNACL_OK) {
    return ret;
  }

  ret = ConvDwIndirectMallocIndirectBuffer(conv_dw);
  if (ret != NNACL_OK) {
    return ret;
  }

  self->thread_nr_ = NNACL_MIN(self->thread_nr_, conv_dw->conv_.compute_.out_h_);
  NNACL_CHECK_ZERO_RETURN_ERR(self->thread_nr_);
  return NNACL_OK;
}

int ConvolutionDepthwiseIndirectPrepare(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < TWO_TENSOR, NNACL_INPUT_TENSOR_ERROR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_OUTPUT_TENSOR_ERROR);
  ConvolutionDepthwiseIndirectStruct *conv_dw = (ConvolutionDepthwiseIndirectStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw);

  ConvBaseUpdateOriginWeightAndBias(&conv_dw->conv_);

  if (self->train_session_) {
    int batch_flag = UP_DIV(conv_dw->conv_.compute_.out_c_, conv_dw->div_flag_);
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv_dw->div_flag_ * batch_flag, conv_dw->conv_.compute_.kernel_hw_, NNACL_ERR);
    int pack_weight_size = conv_dw->div_flag_ * batch_flag * conv_dw->conv_.compute_.kernel_hw_;
    self->work_size_ = pack_weight_size * sizeof(float);
  }

  return ConvBaseInitConvWeightBias(&conv_dw->conv_);
}

int ConvolutionDepthwiseIndirectRelease(KernelBase *self) {
  ConvolutionDepthwiseIndirectStruct *conv_dw = (ConvolutionDepthwiseIndirectStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw);

  if (conv_dw->zero_ptr_ != NULL) {
    self->env_->Free(self->env_->allocator_, conv_dw->zero_ptr_);
    conv_dw->zero_ptr_ = NULL;
  }
  if (conv_dw->indirect_buffer_ != NULL) {
    self->env_->Free(self->env_->allocator_, conv_dw->indirect_buffer_);
    conv_dw->indirect_buffer_ = NULL;
  }
  ConvBaseRelease(&conv_dw->conv_);
  return NNACL_OK;
}

KernelBase *CreateConvDwIndirect(ConvParameter *conv_param) {
  ConvolutionDepthwiseIndirectStruct *conv_dw =
    (ConvolutionDepthwiseIndirectStruct *)malloc(sizeof(ConvolutionDepthwiseIndirectStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(conv_dw);
  memset(conv_dw, 0, sizeof(ConvolutionDepthwiseIndirectStruct));

#ifdef ENABLE_AVX
  conv_dw->div_flag_ = C8NUM;
#else
  conv_dw->div_flag_ = C4NUM;
#endif
  conv_dw->conv_.pack_weight_ = ConvDwIndirectPackWeight;
  conv_dw->conv_.malloc_weight_bias_ = ConvDwIndirectMallocWeightBiasData;

  conv_dw->conv_.base_.Compute = ConvolutionDepthwiseIndirectCompute;
  conv_dw->conv_.base_.Resize = ConvolutionDepthwiseIndirectResize;
  conv_dw->conv_.base_.Prepare = ConvolutionDepthwiseIndirectPrepare;
  conv_dw->conv_.base_.Release = ConvolutionDepthwiseIndirectRelease;

  return (KernelBase *)conv_dw;
}
