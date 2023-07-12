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

#include "nnacl/kernel/convolution_base.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/tensor_c_utils.h"

int ConvBaseUpdateParamInfo(ConvComputeParam *compute, ConvParameter *conv_param) {
  compute->stride_h_ = conv_param->stride_h_;
  compute->stride_w_ = conv_param->stride_w_;
  compute->dilation_h_ = conv_param->dilation_h_;
  compute->dilation_w_ = conv_param->dilation_w_;
  compute->pad_u_ = conv_param->pad_u_;
  compute->pad_d_ = conv_param->pad_d_;
  compute->pad_l_ = conv_param->pad_l_;
  compute->pad_r_ = conv_param->pad_r_;

  compute->in_c_ = conv_param->input_channel_;
  compute->out_c_ = conv_param->output_channel_;

  compute->kernel_h_ = conv_param->kernel_h_;
  compute->kernel_w_ = conv_param->kernel_w_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(compute->kernel_h_, compute->kernel_w_, NNACL_ERR);
  compute->kernel_hw_ = compute->kernel_h_ * compute->kernel_w_;

  return NNACL_OK;
}

int ConvBaseUpdateComputeInfo(ConvolutionBaseStruct *conv) {
  NNACL_CHECK_NULL_RETURN_ERR(conv);
  ConvParameter *conv_param = (ConvParameter *)conv->base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);
  TensorC *input = conv->base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input);
  TensorC *output = conv->base_.out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output);

  conv_param->input_batch_ = GetBatch(input);
  conv_param->input_h_ = GetHeight(input);
  conv_param->input_w_ = GetWidth(input);
  conv_param->input_channel_ = GetChannel(input);
  conv_param->output_batch_ = GetBatch(output);
  conv_param->output_h_ = GetHeight(output);
  conv_param->output_w_ = GetWidth(output);
  conv_param->output_channel_ = GetChannel(output);

  ConvComputeParam *compute = &conv->compute_;
  compute->in_n_ = GetBatch(input);
  compute->in_h_ = GetHeight(input);
  compute->in_w_ = GetWidth(input);
  compute->in_c_ = GetChannel(input);
  NNACL_CHECK_FALSE(compute->in_c_ != conv_param->input_channel_, NNACL_ERR);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(compute->in_h_, compute->in_w_, NNACL_ERR);
  compute->in_hw_ = compute->in_h_ * compute->in_w_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(compute->in_hw_, compute->in_n_, NNACL_ERR);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(compute->in_hw_ * compute->in_n_, compute->in_c_, NNACL_ERR);

  compute->out_n_ = GetBatch(output);
  compute->out_h_ = GetHeight(output);
  compute->out_w_ = GetWidth(output);
  compute->out_c_ = GetChannel(output);
  NNACL_CHECK_FALSE(compute->out_c_ != conv_param->output_channel_, NNACL_ERR);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(compute->out_h_, compute->out_w_, NNACL_ERR);
  compute->out_hw_ = compute->out_h_ * compute->out_w_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(compute->out_hw_, compute->out_n_, NNACL_ERR);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(compute->out_hw_ * compute->out_n_, compute->out_c_, NNACL_ERR);

  return ConvBaseUpdateParamInfo(compute, conv_param);
}

void ConvBaseRelease(ConvolutionBaseStruct *conv) {
  if (!conv->base_.train_session_) {
    if (!conv->is_sharing_pack_) {
      conv->base_.env_->Free(conv->base_.env_->allocator_, conv->packed_weight_);
    } else {
      conv->free_sharing_weight_(conv->shaing_manager_, conv->packed_weight_);
    }
    conv->packed_weight_ = NULL;
  }

  if (conv->bias_data_ != NULL) {
    conv->base_.env_->Free(conv->base_.env_->allocator_, conv->bias_data_);
    conv->bias_data_ = NULL;
  }
}

int ConvBasePrepare(ConvolutionBaseStruct *conv) {
  NNACL_CHECK_FALSE(conv->base_.in_size_ < TWO_TENSOR, NNACL_INPUT_TENSOR_ERROR);
  NNACL_CHECK_FALSE(conv->base_.out_size_ < ONE_TENSOR, NNACL_OUTPUT_TENSOR_ERROR);

  conv->out_format_ = conv->base_.out_[OUTPUT_INDEX]->format_;
  return ConvBaseUpdateComputeInfo(conv);
}

void ConvBaseUpdateOriginWeightAndBias(ConvolutionBaseStruct *conv) {
  NNACL_CHECK_NULL_RETURN_VOID(conv);

  if (conv->base_.in_[SECOND_INPUT]->data_ != NULL) {
    conv->origin_weight_ = conv->base_.in_[SECOND_INPUT]->data_;
  }

  if (conv->base_.in_size_ == THREE_TENSOR && conv->base_.in_[THIRD_INPUT]->data_ != NULL) {
    conv->origin_bias_ = conv->base_.in_[THIRD_INPUT]->data_;
  }
}

int ConvBaseInitConvWeightBias(ConvolutionBaseStruct *conv) {
  if (conv->base_.train_session_) {
    ConvBaseUpdateOriginWeightAndBias(conv);
  }

  /* check weight shape done */
  if (!CheckInferShapeDone(&conv->base_.in_[SECOND_INPUT], ONE_TENSOR, NULL, 0)) {
    return NNACL_OK;
  }

  int ret = conv->malloc_weight_bias_(conv);
  if (ret != NNACL_OK) {
    return ret;
  }

  if (conv->base_.in_size_ == THREE_TENSOR) {
    memcpy(conv->bias_data_, conv->origin_bias_, GetSize(conv->base_.in_[THIRD_INPUT]));
  }

  if (!conv->base_.train_session_) {
    if (conv->weight_is_packed_) {
      return NNACL_OK;
    }
    if (conv->origin_weight_ != NULL) {
      conv->pack_weight_(conv);
    } else {
      conv->is_repack_ = true;
    }
  }
  return NNACL_OK;
}

int ConvBaseCheckResizeValid(ConvolutionBaseStruct *conv) {
  // ===============check in channel================= //
  TensorC *input_tensor = conv->base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input_tensor);
  int resize_in_channel = GetChannel(input_tensor);
  TensorC *filter_tensor = conv->base_.in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(filter_tensor);
  int filter_in_channel = GetChannel(filter_tensor);
  if (filter_in_channel != resize_in_channel) {
    return NNACL_CONVOLUTION_INPUT_CHANNEL_UNMATCH;
  }
  return NNACL_OK;
}

void *ConvBaseGetConvPackWeightData(ConvolutionBaseStruct *conv, int data_size) {
  TensorC *weight_tensor = conv->base_.in_[SECOND_INPUT];
  bool const_fit = weight_tensor->category_ != ConstTensor && weight_tensor->category_ != ConstScalar;
  bool group_fit = ((ConvParameter *)conv->base_.param_)->group_ > 1;
  bool sharing_fit = conv->get_sharing_weight_ == NULL;

  void *data = NULL;
  if (sharing_fit || const_fit || group_fit) {
    if (data_size <= 0) {
      return NULL;
    }
    data = conv->base_.env_->Alloc(conv->base_.env_->allocator_, data_size);
    conv->weight_is_packed_ = false;
    conv->is_sharing_pack_ = false;
  } else {
    data = conv->get_sharing_weight_(conv->shaing_manager_, weight_tensor->data_, data_size, &conv->weight_is_packed_);
  }
  return data;
}

int ConvBaseRepackWeight(ConvolutionBaseStruct *conv) {
  NNACL_CHECK_NULL_RETURN_ERR(conv);

  conv->origin_weight_ = conv->origin_weight_ != NULL ? conv->origin_weight_ : conv->base_.in_[SECOND_INPUT]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(conv->origin_weight_);

  if (conv->packed_weight_ == NULL) {
    int ret = ConvBaseInitConvWeightBias(conv);
    if (ret != NNACL_OK) {
      return ret;
    }
  }

  if (conv->is_repack_ || conv->base_.train_session_) {
    if (conv->base_.train_session_) {
      conv->packed_weight_ = (float *)conv->base_.workspace_;
      memset(conv->packed_weight_, 0, conv->base_.work_size_);
    } else {
      conv->is_repack_ = false;
    }
    conv->pack_weight_(conv);
  }
  return NNACL_OK;
}
