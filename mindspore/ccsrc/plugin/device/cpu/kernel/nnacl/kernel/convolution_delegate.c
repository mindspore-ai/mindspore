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
#include "nnacl/tensor_c_utils.h"
#include "nnacl/base/conv_common_base.h"
#include "nnacl/kernel/group_convolution.h"
#include "nnacl/kernel/convolution_depthwise.h"
#include "nnacl/kernel/convolution_1x1.h"
#include "nnacl/kernel/convolution_im2col.h"
#include "nnacl/kernel/convolution_winograd.h"
#include "nnacl/fp32/conv_winograd_fp32.h"
#include "nnacl/kernel/convolution_depthwise_sw.h"
#ifdef ENABLE_AVX
#include "nnacl/kernel/convolution_sw_1x1.h"
#include "nnacl/kernel/convolution_sw_avx.h"
#include "nnacl/kernel/convolution_depthwise_sw_avx.h"
#endif
#ifdef ENABLE_ARM64
#include "nnacl/kernel/convolution_depthwise_indirect.h"
#include "nnacl/kernel/convolution_sw_arm64.h"
#include "nnacl/fp32/conv_sw_arm64_fp32.h"
#endif
#if defined(ENABLE_ARM) || (defined(ENABLE_SSE) && !defined(ENABLE_AVX))
#include "nnacl/kernel/convolution_depthwise_3x3.h"
#include "nnacl/fp32/conv_depthwise_fp32.h"
#endif

#define MaxDwConvSWSize 32

float *ConvolutionDelegateCopyData(const TensorC *tensor) {
  NNACL_CHECK_NULL_RETURN_NULL(tensor);
  NNACL_CHECK_NULL_RETURN_NULL(tensor->data_);

  float *data = (float *)malloc(GetSize(tensor));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(data);

  (void)memcpy(data, tensor->data_, GetSize(tensor));
  return data;
}

int ConvolutionDelegateGetWeightData(ConvolutionDelegateStruct *convolution_delegate) {
  if (convolution_delegate->conv_.base_.in_[SECOND_INPUT]->data_ == NULL) {
    return NNACL_OK;
  }
  if (convolution_delegate->conv_.infershape_done_) {
    convolution_delegate->origin_weight_ = convolution_delegate->conv_.base_.in_[SECOND_INPUT]->data_;
    NNACL_CHECK_NULL_RETURN_ERR(convolution_delegate->origin_weight_);
    convolution_delegate->need_free_weight_ = false;
    return NNACL_OK;
  }
  convolution_delegate->origin_weight_ =
    ConvolutionDelegateCopyData(convolution_delegate->conv_.base_.in_[SECOND_INPUT]);
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(convolution_delegate->origin_weight_);
  convolution_delegate->need_free_weight_ = true;
  return NNACL_OK;
}

int ConvolutionDelegateGetBiasData(ConvolutionDelegateStruct *convolution_delegate) {
  if (convolution_delegate->conv_.base_.in_size_ != THREE_TENSOR) {
    convolution_delegate->origin_bias_ = NULL;
    convolution_delegate->need_free_bias_ = false;
    return NNACL_OK;
  }

  if (convolution_delegate->conv_.infershape_done_) {
    NNACL_CHECK_NULL_RETURN_ERR(convolution_delegate->conv_.base_.in_[THIRD_INPUT]);
    convolution_delegate->origin_bias_ = convolution_delegate->conv_.base_.in_[THIRD_INPUT]->data_;
    NNACL_CHECK_NULL_RETURN_ERR(convolution_delegate->origin_bias_);
    convolution_delegate->need_free_bias_ = false;
    return NNACL_OK;
  }

  convolution_delegate->origin_bias_ = ConvolutionDelegateCopyData(convolution_delegate->conv_.base_.in_[THIRD_INPUT]);
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

ConvolutionBaseStruct *ConvolutionDelegateConvNC4KernelSelect(ConvolutionDelegateStruct *convolution_delegate) {
  /* runtime nc4hw4 pass
   * arm64: conv1x1 conv_Im2col support nc4
   * Avx: conv_Im2col support nc4
   * */
  ConvParameter *conv_param = (ConvParameter *)convolution_delegate->conv_.base_.param_;
  NNACL_CHECK_NULL_RETURN_NULL(conv_param);

#ifdef ENABLE_ARM64
  if (conv_param->kernel_h_ == 1 && conv_param->kernel_w_ == 1) {
    ConvolutionBaseStruct *conv1x1 = CreateConvolution1x1(conv_param);
    return conv1x1;
  }
#endif

#if defined(ENABLE_ARM64) || defined(ENABLE_AVX)
  ConvolutionBaseStruct *conv_im2col = CreateConvolutionIm2Col(&convolution_delegate->conv_.base_, conv_param);
  return conv_im2col;
#endif

  return NULL;
}

ConvolutionBaseStruct *ConvolutionDelegateConvNHWCKernelSelect(ConvolutionDelegateStruct *convolution_delegate) {
  ConvParameter *conv_param = (ConvParameter *)convolution_delegate->conv_.base_.param_;
  NNACL_CHECK_NULL_RETURN_NULL(conv_param);

  ConvolutionBaseStruct *conv = NULL;

  int out_unit;
  if (CheckIfUseWinograd(&out_unit, conv_param)) {
    conv = CreateConvolutionWinograd(conv_param, out_unit);
  }

#ifdef ENABLE_AVX
  if (conv == NULL && CheckAvxUseSW1x1Conv(conv_param)) {
    conv = CreateConvolutionSW1x1(conv_param, convolution_delegate->input_const_, convolution_delegate->weight_const_);
  }

  if (conv == NULL && CheckAvxUseSWConv(conv_param, convolution_delegate->conv_.base_.thread_nr_)) {
    conv = CreateConvolutionSWAVX(conv_param);
  }
#endif

#ifdef ENABLE_ARM64
  if (conv == NULL && CheckArm64UseSWConv(conv_param)) {
    conv = CreateConvolutionSWARM64(conv_param);
  }
#endif

  if (conv == NULL) {
    if (conv_param->kernel_h_ == 1 && conv_param->kernel_w_ == 1) {
      conv = CreateConvolution1x1(conv_param);
    } else {
      conv = CreateConvolutionIm2Col(&convolution_delegate->conv_.base_, conv_param);
    }
  }
  return conv;
}

ConvolutionBaseStruct *ConvolutionDelegateConvolutionSelect(ConvolutionDelegateStruct *convolution_delegate) {
  ConvolutionBaseStruct *conv;
  if (convolution_delegate->conv_.base_.out_[OUTPUT_INDEX]->format_ == Format_NC4HW4) {
    conv = ConvolutionDelegateConvNC4KernelSelect(convolution_delegate);
  } else {
    conv = ConvolutionDelegateConvNHWCKernelSelect(convolution_delegate);
  }
  if (conv == NULL) {
    return NULL;
  }

  conv->base_.InferShape = convolution_delegate->conv_.base_.InferShape;
  conv->base_.UpdateThread = convolution_delegate->conv_.base_.UpdateThread;
  conv->base_.env_ = convolution_delegate->conv_.base_.env_;
  conv->base_.param_ = convolution_delegate->conv_.base_.param_;
  conv->base_.thread_nr_ = convolution_delegate->conv_.base_.thread_nr_;
  conv->base_.train_session_ = convolution_delegate->conv_.base_.train_session_;
  conv->base_.in_ = convolution_delegate->conv_.base_.in_;
  conv->base_.in_size_ = convolution_delegate->conv_.base_.in_size_;
  conv->base_.out_ = convolution_delegate->conv_.base_.out_;
  conv->base_.out_size_ = convolution_delegate->conv_.base_.out_size_;

  conv->infershape_done_ = convolution_delegate->conv_.infershape_done_;
  conv->shaing_manager_ = convolution_delegate->conv_.shaing_manager_;
  conv->get_sharing_weight_ = convolution_delegate->conv_.get_sharing_weight_;
  conv->free_sharing_weight_ = convolution_delegate->conv_.free_sharing_weight_;
  conv->is_sharing_pack_ = convolution_delegate->conv_.is_sharing_pack_;

  conv->origin_weight_ = convolution_delegate->origin_weight_;
  conv->origin_bias_ = convolution_delegate->origin_bias_;
  return conv;
}

void ConvolutionDelegateFreeCopiedData(ConvolutionDelegateStruct *convolution_delegate) {
  if (convolution_delegate->origin_weight_ != NULL && convolution_delegate->need_free_weight_) {
    free(convolution_delegate->origin_weight_);
  }
  convolution_delegate->origin_weight_ = NULL;
  convolution_delegate->conv_.origin_weight_ = NULL;
  convolution_delegate->need_free_weight_ = false;

  if (convolution_delegate->origin_bias_ != NULL && convolution_delegate->need_free_bias_) {
    free(convolution_delegate->origin_bias_);
  }
  convolution_delegate->origin_bias_ = NULL;
  convolution_delegate->conv_.origin_bias_ = NULL;
  convolution_delegate->need_free_bias_ = false;
}

int ConvolutionDelegateResize(struct KernelBase *self) {
  ConvolutionDelegateStruct *convolution_delegate = (ConvolutionDelegateStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(convolution_delegate);

  if (convolution_delegate->convolution_ == NULL) {
    convolution_delegate->convolution_ = ConvolutionDelegateConvolutionSelect(convolution_delegate);
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(convolution_delegate->convolution_);
    (void)ConvBaseUpdateComputeInfo(convolution_delegate->convolution_);
    int ret = convolution_delegate->convolution_->base_.Prepare(&convolution_delegate->convolution_->base_);
    if (ret != NNACL_OK) {
      return ret;
    }
  }

  (void)ConvBaseUpdateComputeInfo(convolution_delegate->convolution_);
  int ret = convolution_delegate->convolution_->base_.Resize(&convolution_delegate->convolution_->base_);
  if (ret != NNACL_OK) {
    return ret;
  }

  ConvolutionDelegateFreeCopiedData(convolution_delegate);
  return NNACL_OK;
}

int ConvolutionDelegatePrepare(struct KernelBase *self) {
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

  convolution_delegate->input_const_ = IsConst(self->in_[FIRST_INPUT]) && !self->train_session_;
  convolution_delegate->weight_const_ = IsConst(self->in_[SECOND_INPUT]) && !self->train_session_;

  return ConvolutionDelegateGetWeightAndBias(convolution_delegate);
}

int ConvolutionDelegateRelease(struct KernelBase *self) {
  ConvolutionDelegateStruct *convolution_delegate = (ConvolutionDelegateStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(convolution_delegate);
  if (convolution_delegate->convolution_ != NULL) {
    (void)convolution_delegate->convolution_->base_.Release(&convolution_delegate->convolution_->base_);
    free(convolution_delegate->convolution_);
    convolution_delegate->convolution_ = NULL;
  }
  return NNACL_OK;
}

int ConvolutionDelegateCompute(struct KernelBase *self) {
  ConvolutionDelegateStruct *convolution_delegate = (ConvolutionDelegateStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(convolution_delegate);
  NNACL_CHECK_NULL_RETURN_ERR(convolution_delegate->convolution_);

  convolution_delegate->convolution_->base_.workspace_ = convolution_delegate->conv_.base_.workspace_;
  return convolution_delegate->convolution_->base_.Compute(&convolution_delegate->convolution_->base_);
}

KernelBase *CreateConvlutionDelegate(ConvParameter *conv_param) {
  ConvolutionDelegateStruct *delegate = (ConvolutionDelegateStruct *)malloc(sizeof(ConvolutionDelegateStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(delegate);
  memset(delegate, 0, sizeof(ConvolutionDelegateStruct));
  delegate->conv_.base_.Prepare = ConvolutionDelegatePrepare;
  delegate->conv_.base_.Resize = ConvolutionDelegateResize;
  delegate->conv_.base_.Release = ConvolutionDelegateRelease;
  delegate->conv_.base_.Compute = ConvolutionDelegateCompute;
  return (KernelBase *)delegate;
}

KernelBase *CreateConvolutionDepthwise(ConvParameter *conv_param) {
  NNACL_CHECK_NULL_RETURN_NULL(conv_param);
  KernelBase *kernel = NULL;

  if (conv_param->dynamic_shape_) {
    kernel = CreateConvDw(conv_param);
    if (kernel != NULL) {
      return kernel;
    }
  }

#ifdef ENABLE_AVX
  kernel = CreateConvDwSWAVX(conv_param);
  if (kernel != NULL) {
    return kernel;
  }
#endif

#if defined(ENABLE_ARM) || (defined(ENABLE_SSE) && !defined(ENABLE_AVX))
  if (CheckConvDw1DWinograd(conv_param, conv_param->thread_num_)) {
    kernel = CreateConvDw3x3(conv_param);
    if (kernel != NULL) {
      return kernel;
    }
  }
#endif

#ifdef ENABLE_ARM64
  if (CheckConvDwUseIndirectBuffer(conv_param)) {
    kernel = CreateConvDwIndirect(conv_param);
    if (kernel != NULL) {
      return kernel;
    }
  }
#endif

  if (conv_param->input_channel_ < MaxDwConvSWSize) {
    kernel = CreateConvDwSW(conv_param);
    if (kernel != NULL) {
      return kernel;
    }
  }

  kernel = CreateConvDw(conv_param);
  return kernel;
}

KernelBase *CreateConv2DFusion(OpParameter *param, int data_type) {
  ConvParameter *conv_param = (ConvParameter *)param;
  conv_param->thread_num_ = param->thread_num_;
  KernelBase *kernel;
  if (conv_param->group_ == 1) {
    kernel = CreateConvlutionDelegate(conv_param);
  } else if (conv_param->group_ == conv_param->input_channel_ && conv_param->group_ == conv_param->output_channel_) {
    kernel = CreateConvolutionDepthwise(conv_param);
  } else {
    kernel = CreateGroupConvolution(conv_param, data_type);
  }

  if (kernel == NULL) {
    return NULL;
  }

  ConvolutionBaseStruct *conv = (ConvolutionBaseStruct *)kernel;
  (void)ConvBaseUpdateParamInfo(&conv->compute_, conv_param);

  return kernel;
}

REG_KERNEL_CREATOR(PrimType_Conv2DFusion, kNumberTypeFloat32, CreateConv2DFusion)
