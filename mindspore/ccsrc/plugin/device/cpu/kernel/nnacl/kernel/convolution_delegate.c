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

#include "nnacl/kernel/convolution_delegate.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/base/conv_common_base.h"
#include "nnacl/kernel/group_convolution.h"
#include "nnacl/kernel/convolution_depthwise.h"
#include "nnacl/kernel/convolution_1x1.h"
#include "nnacl/kernel/convolution_im2col.h"
#include "nnacl/kernel/convolution_winograd.h"
#ifdef ENABLE_AVX
#include "nnacl/kernel/convolution_sw_1x1.h"
#include "nnacl/kernel/convolution_sw_avx.h"
#endif

float *ConvolutionDelegateCopyData(const TensorC *tensor) {
  NNACL_CHECK_NULL_RETURN_NULL(tensor);
  NNACL_CHECK_NULL_RETURN_NULL(tensor->data_);

  float *data = (float *)malloc(GetSize(tensor));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(data);

  (void)memcpy(data, tensor->data_, GetSize(tensor));
  return data;
}

int ConvolutionDelegateGetWeightData(ConvolutionDelegateStruct *convolution_delegate) {
  if (convolution_delegate->base_.in_[SECOND_INPUT]->data_ == NULL) {
    return NNACL_OK;
  }
  if (convolution_delegate->infershape_done_) {
    convolution_delegate->origin_weight_ = convolution_delegate->base_.in_[SECOND_INPUT]->data_;
    NNACL_CHECK_NULL_RETURN_ERR(convolution_delegate->origin_weight_);
    convolution_delegate->need_free_weight_ = false;
    return NNACL_OK;
  }
  convolution_delegate->origin_weight_ = ConvolutionDelegateCopyData(convolution_delegate->base_.in_[SECOND_INPUT]);
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(convolution_delegate->origin_weight_);
  convolution_delegate->need_free_weight_ = true;
  return NNACL_OK;
}

int ConvolutionDelegateGetBiasData(ConvolutionDelegateStruct *convolution_delegate) {
  if (convolution_delegate->base_.in_size_ != THREE_TENSOR) {
    convolution_delegate->origin_bias_ = NULL;
    convolution_delegate->need_free_bias_ = false;
    return NNACL_OK;
  }

  if (convolution_delegate->infershape_done_) {
    NNACL_CHECK_NULL_RETURN_ERR(convolution_delegate->base_.in_[THIRD_INPUT]);
    convolution_delegate->origin_bias_ = convolution_delegate->base_.in_[THIRD_INPUT]->data_;
    NNACL_CHECK_NULL_RETURN_ERR(convolution_delegate->origin_bias_);
    convolution_delegate->need_free_bias_ = false;
    return NNACL_OK;
  }

  convolution_delegate->origin_bias_ = ConvolutionDelegateCopyData(convolution_delegate->base_.in_[THIRD_INPUT]);
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(convolution_delegate->origin_bias_);
  convolution_delegate->need_free_bias_ = true;
  return NNACL_OK;
}

int ConvolutionDelegateGetWeightAndBias(ConvolutionDelegateStruct *convolution_delegate) {
  int ret = ConvolutionDelegateGetWeightData(convolution_delegate);
  if (ret != NNACL_OK) {
    return ret;
  }

  return ConvolutionDelegateGetBiasData(convolution_delegate);
}

void ConvolutionDelegateSetInputOutputShapeInfo(ConvolutionDelegateStruct *convolution_delegate) {
  NNACL_CHECK_NULL_RETURN_VOID(convolution_delegate);
  ConvParameter *conv_param = (ConvParameter *)convolution_delegate->base_.param_;
  NNACL_CHECK_NULL_RETURN_VOID(conv_param);
  TensorC *input = convolution_delegate->base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_VOID(input);
  TensorC *output = convolution_delegate->base_.out_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_VOID(output);

  conv_param->input_batch_ = GetBatch(input);
  conv_param->input_h_ = GetHeight(input);
  conv_param->input_w_ = GetWidth(input);
  conv_param->input_channel_ = GetChannel(input);
  conv_param->output_batch_ = GetBatch(output);
  conv_param->output_h_ = GetHeight(output);
  conv_param->output_w_ = GetWidth(output);
  conv_param->output_channel_ = GetChannel(output);
  return;
}

ConvolutionBaseStruct *ConvolutionDelegateConvNC4KernelSelect(ConvolutionDelegateStruct *convolution_delegate) {
  /* runtime nc4hw4 pass
   * arm64: conv1x1 conv_Im2col support nc4
   * Avx: conv_Im2col support nc4
   * */
  ConvParameter *conv_param = (ConvParameter *)convolution_delegate->base_.param_;
  NNACL_CHECK_NULL_RETURN_NULL(conv_param);

#ifdef ENABLE_ARM64
  if (conv_param->kernel_h_ == 1 && conv_param->kernel_w_ == 1) {
    ConvolutionBaseStruct *conv1x1 = CreateConvolution1x1(conv_param);
    return conv1x1;
  }
#endif

#if defined(ENABLE_ARM64) || defined(ENABLE_AVX)
  ConvolutionBaseStruct *conv_im2col = CreateConvolutionIm2Col(conv_param);
  return conv_im2col;
#endif

  return NULL;
}

ConvolutionBaseStruct *ConvolutionDelegateConvNHWCKernelSelect(ConvolutionDelegateStruct *convolution_delegate) {
  ConvParameter *conv_param = (ConvParameter *)convolution_delegate->base_.param_;
  NNACL_CHECK_NULL_RETURN_NULL(conv_param);

  ConvolutionBaseStruct *conv = NULL;

  int out_unit;
  if (CheckIfUseWinograd(&out_unit, conv_param)) {
    conv = CreateConvolutionWinograd(conv_param);
  }

#ifdef ENABLE_AVX
  if (conv == NULL && CheckAvxUseSW1x1Conv(conv_param)) {
    conv = CreateConvolutionSW1x1(conv_param);
  }

  if (conv == NULL && CheckAvxUseSWConv(conv_param)) {
    conv = CreateConvolutionSWAVX(conv_param);
  }
#endif

  if (conv == NULL) {
    if (conv_param->kernel_h_ == 1 && conv_param->kernel_w_ == 1) {
      conv = CreateConvolution1x1(conv_param);
    } else {
      conv = CreateConvolutionIm2Col(conv_param);
    }
  }
  return conv;
}

ConvolutionBaseStruct *ConvolutionDelegateConvolutionSelect(ConvolutionDelegateStruct *convolution_delegate) {
  ConvolutionBaseStruct *conv;
  if (convolution_delegate->base_.out_[OUTPUT_INDEX]->format_ == NC4HW4) {
    conv = ConvolutionDelegateConvNC4KernelSelect(convolution_delegate);
  } else {
    conv = ConvolutionDelegateConvNHWCKernelSelect(convolution_delegate);
  }
  if (conv == NULL) {
    return NULL;
  }

  conv->base_.infershape = convolution_delegate->base_.infershape;
  conv->base_.env_ = convolution_delegate->base_.env_;
  conv->base_.param_ = convolution_delegate->base_.param_;
  conv->base_.thread_nr_ = convolution_delegate->base_.thread_nr_;
  conv->base_.train_session_ = convolution_delegate->base_.train_session_;
  conv->base_.in_ = convolution_delegate->base_.in_;
  conv->base_.in_size_ = convolution_delegate->base_.in_size_;
  conv->base_.out_ = convolution_delegate->base_.out_;
  conv->base_.out_size_ = convolution_delegate->base_.out_size_;
  conv->base_.update_thread_ = convolution_delegate->base_.update_thread_;

  int ret = conv->base_.prepare(&conv->base_);
  if (ret != NNACL_OK) {
    conv->base_.release(&conv->base_);
    free(conv);
    conv = NULL;
  }

  return conv;
}

void ConvolutionDelegateFreeCopiedData(ConvolutionDelegateStruct *convolution_delegate) {
  if (convolution_delegate->origin_weight_ != NULL && convolution_delegate->need_free_weight_) {
    free(convolution_delegate->origin_weight_);
  }
  convolution_delegate->origin_weight_ = NULL;
  convolution_delegate->need_free_weight_ = false;

  if (convolution_delegate->origin_bias_ != NULL && convolution_delegate->need_free_bias_) {
    free(convolution_delegate->origin_bias_);
  }
  convolution_delegate->origin_bias_ = NULL;
  convolution_delegate->need_free_bias_ = false;
}

int convolution_delegate_resize(struct KernelBase *self) {
  ConvolutionDelegateStruct *convolution_delegate = (ConvolutionDelegateStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(convolution_delegate);

  ConvolutionDelegateSetInputOutputShapeInfo(convolution_delegate);

  if (convolution_delegate->convolution_ == NULL) {
    convolution_delegate->convolution_ = ConvolutionDelegateConvolutionSelect(convolution_delegate);
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(convolution_delegate->convolution_);
  }

  ConvolutionDelegateFreeCopiedData(convolution_delegate);
  return convolution_delegate->convolution_->base_.resize(&convolution_delegate->convolution_->base_);
}

int convolution_delegate_prepare(struct KernelBase *self) {
  ConvolutionDelegateStruct *convolution_delegate = (ConvolutionDelegateStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(convolution_delegate);

  NNACL_CHECK_FALSE(self->in_size_ < TWO_TENSOR, NNACL_INPUT_TENSOR_ERROR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_OUTPUT_TENSOR_ERROR);
  NNACL_CHECK_NULL_RETURN_ERR(self->in_[SECOND_INPUT]);

  NNACL_CHECK_FALSE(self->in_[SECOND_INPUT]->data_type_ != kNumberTypeFloat32 &&
                      self->in_[SECOND_INPUT]->data_type_ != kNumberTypeFloat16,
                    NNACL_CONVOLUTION_WEIGHT_DATATYPE_INVALID);
  NNACL_CHECK_FALSE(self->in_size_ == THREE_TENSOR && self->in_[THIRD_INPUT] != NULL &&
                      self->in_[THIRD_INPUT]->data_type_ != kNumberTypeFloat32,
                    NNACL_CONVOLUTION_BIAS_DATATYPE_INVALID);

  convolution_delegate->infershape_done_ = CheckInferShapeDone(self->in_, self->in_size_, self->out_, self->out_size_);

  return ConvolutionDelegateGetWeightAndBias(convolution_delegate);
}

int convolution_delegate_release(struct KernelBase *self) {
  ConvolutionDelegateStruct *convolution_delegate = (ConvolutionDelegateStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(convolution_delegate);
  int ret = NNACL_OK;
  if (convolution_delegate->convolution_ != NULL) {
    ret = convolution_delegate->convolution_->base_.release(&convolution_delegate->convolution_->base_);
    free(convolution_delegate->convolution_);
    convolution_delegate->convolution_ = NULL;
  }
  return ret;
}

int convolution_delegate_compute(struct KernelBase *self) {
  ConvolutionDelegateStruct *convolution_delegate = (ConvolutionDelegateStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(convolution_delegate);
  NNACL_CHECK_NULL_RETURN_ERR(convolution_delegate->convolution_);

  convolution_delegate->convolution_->base_.workspace_ = convolution_delegate->base_.workspace_;
  return convolution_delegate->convolution_->base_.compute(&convolution_delegate->convolution_->base_);
}

KernelBase *CreateConvlutionDelegate(ConvParameter *conv_param) {
  ConvolutionDelegateStruct *delegate = (ConvolutionDelegateStruct *)malloc(sizeof(ConvolutionDelegateStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(delegate);
  memset(delegate, 0, sizeof(ConvolutionDelegateStruct));

  delegate->base_.prepare = convolution_delegate_prepare;
  delegate->base_.resize = convolution_delegate_resize;
  delegate->base_.release = convolution_delegate_release;
  delegate->base_.compute = convolution_delegate_compute;
  return (KernelBase *)delegate;
}

KernelBase *CreateConv2DFusion(OpParameter *param, int data_type) {
  ConvParameter *conv_param = (ConvParameter *)param;
  KernelBase *kernel;
  if (conv_param->group_ == 1) {
    kernel = CreateConvlutionDelegate(conv_param);
  } else if (conv_param->group_ == conv_param->input_channel_ && conv_param->group_ == conv_param->output_channel_) {
    kernel = CreateConvolutionDepthwise(conv_param);
  } else {
    kernel = CreateGroupConvolution(conv_param);
  }
  return kernel;
}

REG_KERNEL_CREATOR(PrimType_Conv2DFusion, kNumberTypeFloat32, CreateConv2DFusion);
