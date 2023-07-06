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

#if defined(ENABLE_AVX) || defined(ENABLE_ARM64)
#include "nnacl/kernel/convolution_slidewindow.h"
#include "nnacl/fp32/conv_depthwise_fp32.h"
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/tensor_c.h"
#include "nnacl/tensor_c_utils.h"

int ConvSWInitTmpBuffer(ConvolutionSWStruct *conv_sw) {
  TensorC *input_tensor = conv_sw->conv_.base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input_tensor);
  float *input_data = (float *)input_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(input_data);
  ConvComputeParam *compute = &conv_sw->conv_.compute_;
  NNACL_CHECK_NULL_RETURN_ERR(compute);

  if (conv_sw->ic_res_ != 0 && compute->kernel_h_ == 1 && compute->kernel_w_ == 1) {
    int ic_block_num = UP_DIV(compute->in_c_, conv_sw->in_tile_);
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(compute->in_n_, compute->in_hw_, NNACL_ERR);
    int input_bhw = compute->in_n_ * conv_sw->conv_.compute_.in_hw_;
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(input_bhw, ic_block_num * conv_sw->in_tile_, NNACL_ERR);

    conv_sw->input_data_ = (float *)conv_sw->conv_.base_.env_->Alloc(
      conv_sw->conv_.base_.env_->allocator_, input_bhw * ic_block_num * conv_sw->in_tile_ * sizeof(float));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv_sw->input_data_);

    PackNHWCToNHWCXFp32(input_data, conv_sw->input_data_, compute->in_n_, compute->in_hw_, compute->in_c_,
                        conv_sw->oc_tile_);
  } else {
    conv_sw->input_data_ = input_data;
  }

  float *out_data = (float *)conv_sw->conv_.base_.out_[OUTPUT_INDEX]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(out_data);
  if (conv_sw->oc_res_ == 0) {  // not need to malloc dst
    conv_sw->output_data_ = out_data;
  } else {  // need to malloc dst to align block
    int oc_block_num = UP_DIV(compute->out_c_, conv_sw->oc_tile_);
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(compute->out_n_, compute->out_hw_, NNACL_ERR);
    int output_bhw = compute->out_n_ * compute->out_hw_;
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(output_bhw, oc_block_num * conv_sw->oc_tile_, NNACL_ERR);
    conv_sw->output_data_ = (float *)conv_sw->conv_.base_.env_->Alloc(
      conv_sw->conv_.base_.env_->allocator_, output_bhw * oc_block_num * conv_sw->oc_tile_ * sizeof(float));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv_sw->output_data_);
  }

  return NNACL_OK;
}

void ConvSWFreeTmpBuffer(ConvolutionSWStruct *conv_sw) {
  ConvParameter *conv_param = (ConvParameter *)conv_sw->conv_.base_.param_;
  NNACL_CHECK_NULL_RETURN_VOID(conv_param);

  if (conv_sw->output_data_ != NULL && conv_sw->oc_res_ != 0) {
    conv_sw->conv_.base_.env_->Free(conv_sw->conv_.base_.env_->allocator_, conv_sw->output_data_);
    conv_sw->output_data_ = NULL;
  }
  if (conv_sw->input_data_ != NULL && conv_sw->ic_res_ != 0 && conv_param->kernel_w_ == 1 &&
      conv_param->kernel_h_ == 1) {
    conv_sw->conv_.base_.env_->Free(conv_sw->conv_.base_.env_->allocator_, conv_sw->input_data_);
    conv_sw->input_data_ = NULL;
  }
}

void ConvSWPackWeight(ConvolutionBaseStruct *conv) {
  ConvolutionSWStruct *conv_sw = (ConvolutionSWStruct *)conv;
  NNACL_CHECK_NULL_RETURN_VOID(conv_sw);
  TensorC *filter_tensor = conv->base_.in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_VOID(filter_tensor);

  int input_channel = GetChannel(filter_tensor);
  int output_channel = GetBatch(filter_tensor);
  int kernel_h = GetHeight(filter_tensor);
  int kernel_w = GetWidth(filter_tensor);

  int oc_block_num = UP_DIV(output_channel, conv_sw->oc_tile_);
  void *origin_weight = (conv->base_.train_session_) ? filter_tensor->data_ : conv->origin_weight_;
  NNACL_CHECK_NULL_RETURN_VOID(origin_weight);
  PackNHWCToNXHWCXFp32(kernel_h, kernel_w, output_channel, oc_block_num, input_channel, (float *)conv->packed_weight_,
                       (float *)origin_weight);
}

int ConvSWMallocWeightBiasData(ConvolutionBaseStruct *conv) {
  ConvolutionSWStruct *conv_sw = (ConvolutionSWStruct *)conv;
  NNACL_CHECK_NULL_RETURN_ERR(conv_sw);
  ConvParameter *conv_param = (ConvParameter *)conv->base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);
  TensorC *filter_tensor = conv->base_.in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(filter_tensor);

  int input_channel = GetChannel(filter_tensor);
  int output_channel = GetBatch(filter_tensor);
  int kernel_h = GetHeight(filter_tensor);
  int kernel_w = GetWidth(filter_tensor);

  conv_param->input_channel_ = input_channel;
  conv_param->output_channel_ = output_channel;
  int kernel_plane = kernel_h * kernel_w;
  int oc_block_num = UP_DIV(output_channel, conv_sw->oc_tile_);
  int pack_weight_size = oc_block_num * conv_sw->oc_tile_ * input_channel * kernel_plane;
  if (!conv_sw->conv_.base_.train_session_) {
    conv_sw->conv_.packed_weight_ = ConvBaseGetConvPackWeightData(conv, pack_weight_size * sizeof(float));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv_sw->conv_.packed_weight_);
  }

  if (conv_sw->conv_.base_.in_size_ == THREE_TENSOR) {
    int malloc_size = oc_block_num * conv_sw->oc_tile_ * sizeof(float);
    conv->bias_data_ = conv->base_.env_->Alloc(conv->base_.env_->allocator_, malloc_size);
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv->bias_data_);
    memset(conv->bias_data_, 0, oc_block_num * conv_sw->oc_tile_ * sizeof(float));
  }
  return NNACL_OK;
}

int ConvSWImpl(void *cdata, int task_id, float l, float r) {
  ConvolutionSWStruct *conv_sw = (ConvolutionSWStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(conv_sw);
  return conv_sw->conv_.run_impl_(&conv_sw->conv_, task_id);
}

int ConvolutionSWCompute(KernelBase *self) {
  ConvolutionSWStruct *conv_sw = (ConvolutionSWStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_sw);

  int ret = ConvSWInitTmpBuffer(conv_sw);
  if (ret != NNACL_OK) {
    ConvSWFreeTmpBuffer(conv_sw);
    return ret;
  }

  ret = ConvBaseRepackWeight(&conv_sw->conv_);
  if (ret != NNACL_OK) {
    ConvSWFreeTmpBuffer(conv_sw);
    return ret;
  }

  ret = self->env_->ParallelLaunch(self->env_->thread_pool_, ConvSWImpl, self, self->thread_nr_);
  if (ret != NNACL_OK) {
    ConvSWFreeTmpBuffer(conv_sw);
    return ret;
  }

  if (conv_sw->oc_res_ != 0) {
    ConvParameter *conv_param = (ConvParameter *)self->param_;
    NNACL_CHECK_NULL_RETURN_ERR(conv_param);
    float *out_data = (float *)self->out_[OUTPUT_INDEX]->data_;
    NNACL_CHECK_NULL_RETURN_ERR(out_data);
    PackNHWCXToNHWCFp32(conv_sw->output_data_, out_data, conv_param->output_batch_,
                        conv_param->output_h_ * conv_param->output_w_, conv_param->output_channel_, conv_sw->oc_tile_);
  }

  ConvSWFreeTmpBuffer(conv_sw);
  return NNACL_OK;
}

int ConvolutionSWRelease(KernelBase *self) {
  ConvolutionBaseStruct *conv = (ConvolutionBaseStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv);
  ConvBaseRelease(conv);
  return NNACL_OK;
}

int ConvolutionSWResize(KernelBase *self) {
  ConvolutionSWStruct *conv_sw = (ConvolutionSWStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_sw);
  ConvParameter *conv_param = (ConvParameter *)self->param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);

  int ret = ConvBaseCheckResizeValid(&conv_sw->conv_);
  if (ret != NNACL_OK) {
    return ret;
  }

  ret = ConvBasePrepare(&conv_sw->conv_);
  if (ret != NNACL_OK) {
    return ret;
  }

  InitSlidingParamConv(&conv_sw->sw_param_, conv_param, conv_sw->in_tile_, conv_sw->oc_tile_);
  return NNACL_OK;
}

int ConvolutionSWPrepare(KernelBase *self) {
  ConvolutionSWStruct *conv_sw = (ConvolutionSWStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_sw);

  conv_sw->conv_.init_global_variable_(&conv_sw->conv_);

  if (self->train_session_) {
    TensorC *filter_tensor = self->in_[SECOND_INPUT];
    NNACL_CHECK_NULL_RETURN_ERR(filter_tensor);
    NNACL_CHECK_FALSE(filter_tensor->shape_size_ != DIMENSION_4D, NNACL_CONVOLUTION_WEIGHT_SHAPE_INVALID);

    int input_channel = GetChannel(filter_tensor);
    int output_channel = GetBatch(filter_tensor);
    int kernel_h = GetHeight(filter_tensor);
    int kernel_w = GetWidth(filter_tensor);

    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(kernel_h, kernel_w, NNACL_ERR);
    int kernel_hw = kernel_h * kernel_w;
    int oc_block_num = UP_DIV(output_channel, conv_sw->oc_tile_);
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(input_channel, kernel_hw, NNACL_ERR);
    int kernel_chw = input_channel * kernel_hw;
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(oc_block_num * conv_sw->oc_tile_, kernel_chw, NNACL_ERR);
    int pack_weight_size = oc_block_num * conv_sw->oc_tile_ * kernel_chw;

    conv_sw->conv_.base_.work_size_ = pack_weight_size * sizeof(float);
  }

  return ConvBaseInitConvWeightBias(&conv_sw->conv_);
}
#endif
