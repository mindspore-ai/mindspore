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

void ConvBaseRelease(ConvolutionBaseStruct *conv) {
  if (!conv->base_.train_session_) {
    if (!conv->is_sharing_pack_) {
      conv->base_.env_->free(conv->base_.env_->allocator_, conv->packed_weight_);
    } else {
      conv->free_by_sharing_weight_(conv->pack_weight_manager_, conv->packed_weight_);
    }
    conv->packed_weight_ = NULL;
  }

  if (conv->bias_data_ != NULL) {
    conv->base_.env_->free(conv->base_.env_->allocator_, conv->bias_data_);
    conv->bias_data_ = NULL;
  }
}

int ConvBasePrepare(ConvolutionBaseStruct *conv) {
  NNACL_CHECK_FALSE(conv->base_.in_size_ < TWO_TENSOR, NNACL_INPUT_TENSOR_ERROR);
  NNACL_CHECK_FALSE(conv->base_.out_size_ < ONE_TENSOR, NNACL_OUTPUT_TENSOR_ERROR);

  TensorC *input = conv->base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input);
  NNACL_CHECK_FALSE(input->shape_size_ != DIMENSION_4D, NNACL_INPUT_TENSOR_ERROR);
  conv->input_b_ = GetBatch(input);
  conv->input_h_ = GetHeight(input);
  conv->input_w_ = GetWidth(input);
  conv->input_c_ = GetChannel(input);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv->input_h_, conv->input_w_, NNACL_CONVOLUTION_INPUT_HW_OVERFLOW);

  TensorC *output = conv->base_.out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output);
  NNACL_CHECK_FALSE(output->shape_size_ != DIMENSION_4D, NNACL_OUTPUT_TENSOR_ERROR);
  conv->output_b_ = GetBatch(output);
  conv->output_h_ = GetHeight(output);
  conv->output_w_ = GetWidth(output);
  conv->output_c_ = GetChannel(output);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv->output_h_, conv->output_w_, NNACL_CONVOLUTION_INPUT_HW_OVERFLOW);

  TensorC *filter = conv->base_.in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(filter);
  conv->kernel_h_ = GetHeight(filter);
  conv->kernel_w_ = GetWidth(filter);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv->kernel_h_, conv->kernel_w_, NNACL_CONVOLUTION_KERNEL_HW_OVERFLOW);

  return NNACL_OK;
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

  if (!conv->infershape_done_) {
    return NNACL_OK;
  }

  int ret = conv->malloc_weight_bias_(conv);
  if (ret != NNACL_OK) {
    return ret;
  }

  if (conv->base_.in_size_ == THREE_TENSOR) {
    TensorC *bias_tensor = conv->base_.in_[THIRD_INPUT];
    NNACL_CHECK_NULL_RETURN_ERR(bias_tensor);
    NNACL_CHECK_NULL_RETURN_ERR(bias_tensor->data_);
    NNACL_CHECK_FALSE(GetSize(bias_tensor) == 0, NNACL_INPUT_TENSOR_ERROR);
    memcpy(conv->bias_data_, bias_tensor->data_, GetSize(bias_tensor));
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
  void *data = NULL;
  ConvParameter *conv_param = (ConvParameter *)conv->base_.param_;
  if (conv_param->group_ > 1 || (conv->base_.in_[SECOND_INPUT]->category_ != ConstTensor &&
                                 conv->base_.in_[SECOND_INPUT]->category_ != ConstScalar)) {
    if (data_size <= 0) {
      return NULL;
    }
    data = conv->base_.env_->alloc(conv->base_.env_->allocator_, data_size);
    conv->weight_is_packed_ = false;
    conv->is_sharing_pack_ = false;
  } else {
    data = conv->get_pack_data_by_sharing_weight_(conv->pack_weight_manager_, conv->base_.in_[FIRST_INPUT]->data_,
                                                  data_size, &conv->weight_is_packed_);
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
