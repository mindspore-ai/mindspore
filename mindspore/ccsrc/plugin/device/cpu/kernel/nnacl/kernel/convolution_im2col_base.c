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

#include "nnacl/kernel/convolution_im2col_base.h"
#include "nnacl/kernel/convolution_base.h"
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/fp32/conv_common_fp32.h"

int ConvIm2ColBaseImpl(void *cdata, int task_id, float l, float r) {
  ConvolutionBaseStruct *conv = (ConvolutionBaseStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(cdata);
  return conv->run_impl_(conv, task_id);
}

int ConvIm2ColBaseRunImpl(ConvolutionBaseStruct *conv, int task_id) {
  ConvolutionIm2ColBaseStruct *conv_im2col = (ConvolutionIm2ColBaseStruct *)conv;
  NNACL_CHECK_NULL_RETURN_ERR(conv_im2col);
  float *ori_input_data = (float *)conv->base_.in_[FIRST_INPUT]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(ori_input_data);
  ConvParameter *conv_param = (ConvParameter *)conv->base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);

  if (conv->use_batch_cut_flag_) {
    ConvFp32CutByBatch(ori_input_data, conv_im2col->packed_input_, (float *)conv->packed_weight_,
                       (float *)conv->bias_data_, conv_im2col->col_major_input_, conv_im2col->tmp_output_, task_id,
                       conv_param);
  } else {
    ConvFp32(ori_input_data, conv_im2col->packed_input_, (float *)conv->packed_weight_, (float *)conv->bias_data_,
             conv_im2col->col_major_input_, conv_im2col->tmp_output_, task_id, conv_param);
  }
  return NNACL_OK;
}

int ConvIm2ColBaseMallocWeightBiasData(ConvolutionBaseStruct *conv) {
  ConvolutionIm2ColBaseStruct *conv_im2col = (ConvolutionIm2ColBaseStruct *)conv;
  NNACL_CHECK_NULL_RETURN_ERR(conv_im2col);

  size_t oc_block_num = UP_ROUND(conv->output_c_, conv_im2col->oc_tile_);
  size_t kernel_plane = conv->kernel_h_ * conv->kernel_w_;
  size_t pack_weight_size = oc_block_num * conv->input_c_ * kernel_plane;
  if (!conv->base_.train_session_) {
    NNACL_CHECK_MALLOC_SIZE(pack_weight_size * sizeof(float));
    conv->packed_weight_ = ConvBaseGetConvPackWeightData(conv, pack_weight_size * sizeof(float));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv->packed_weight_);
  }

  if (conv->bias_data_ == NULL) {
    NNACL_CHECK_MALLOC_SIZE(oc_block_num * sizeof(float));
    conv->bias_data_ = conv->base_.env_->alloc(conv->base_.env_->allocator_, oc_block_num * sizeof(float));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv->bias_data_);
  }
  memset(conv->bias_data_, 0, oc_block_num * sizeof(float));
  return NNACL_OK;
}

int ConvIm2ColBaseUpdateThreadNumProcess(KernelBase *self, int32_t kernel_type, int64_t per_unit_load_num,
                                         int64_t per_unit_store_num, int64_t unit_num) {
  ConvolutionIm2ColBaseStruct *conv_im2col = (ConvolutionIm2ColBaseStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_im2col);

  if (conv_im2col->conv_.input_b_ % self->thread_nr_ == 0) {
    conv_im2col->conv_.use_batch_cut_flag_ = true;
    return NNACL_OK;
  } else {
    conv_im2col->conv_.use_batch_cut_flag_ = false;
  }

  int output_hw = conv_im2col->conv_.output_h_ * conv_im2col->conv_.output_w_;
  self->thread_nr_ = MSMIN(UP_DIV(UP_DIV(output_hw, conv_im2col->row_tile_), ConvMinBlock), self->thread_nr_);
  return NNACL_OK;
}

void ConvIm2ColBaseFreeTmpBuffer(ConvolutionIm2ColBaseStruct *conv_im2col) {
  ExecEnv *env = conv_im2col->conv_.base_.env_;
  NNACL_CHECK_NULL_RETURN_VOID(env);

  if (conv_im2col->packed_input_ != NULL) {
    env->free(env->allocator_, conv_im2col->packed_input_);
    conv_im2col->packed_input_ = NULL;
  }
  if (conv_im2col->col_major_input_ != NULL) {
    env->free(env->allocator_, conv_im2col->col_major_input_);
    conv_im2col->col_major_input_ = NULL;
  }
  if (conv_im2col->output_need_align_ && conv_im2col->tmp_output_ != NULL) {
    env->free(env->allocator_, conv_im2col->tmp_output_);
    conv_im2col->tmp_output_ = NULL;
    conv_im2col->output_need_align_ = false;
  }
}

int ConvIm2ColBaseInitTmpBuffer(ConvolutionIm2ColBaseStruct *conv_im2col) {
  ConvolutionBaseStruct *conv = (ConvolutionBaseStruct *)conv_im2col;
  TensorC *out_tensor = conv_im2col->conv_.base_.out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(out_tensor);
  NNACL_CHECK_NULL_RETURN_ERR(out_tensor->data_);

  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv->kernel_h_, conv->kernel_w_, NNACL_ERR);
  int kernel_hw = conv->kernel_h_ * conv->kernel_w_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(kernel_hw, conv->input_c_, NNACL_ERR);
  int kernel_chw = kernel_hw * conv->input_c_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(kernel_chw, conv->base_.thread_nr_, NNACL_ERR);
  int total_kernel_chw = kernel_chw * conv->base_.thread_nr_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(total_kernel_chw, conv_im2col->row_tile_, NNACL_ERR);
  int unit_size = total_kernel_chw * conv_im2col->row_tile_;

  if (conv_im2col->packed_input_ != NULL) {
    conv->base_.env_->free(conv->base_.env_->allocator_, conv_im2col->packed_input_);
    conv_im2col->packed_input_ = NULL;
  }
  conv_im2col->packed_input_ =
    (float *)conv->base_.env_->alloc(conv->base_.env_->allocator_, unit_size * sizeof(float));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv_im2col->packed_input_);

  if (conv_im2col->col_major_input_ != NULL) {
    conv->base_.env_->free(conv->base_.env_->allocator_, conv_im2col->col_major_input_);
    conv_im2col->col_major_input_ = NULL;
  }
  conv_im2col->col_major_input_ =
    (float *)conv->base_.env_->alloc(conv->base_.env_->allocator_, unit_size * sizeof(float));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv_im2col->col_major_input_);

  return NNACL_OK;
}

void ConvIm2ColBasePackWeight(ConvolutionBaseStruct *conv) {
  void *origin_weight = (conv->base_.train_session_) ? conv->base_.in_[SECOND_INPUT]->data_ : conv->origin_weight_;
  NNACL_CHECK_NULL_RETURN_VOID(origin_weight);

  ConvolutionIm2ColBaseStruct *conv_im2col = (ConvolutionIm2ColBaseStruct *)conv;
  NNACL_CHECK_NULL_RETURN_VOID(conv_im2col->row_major_to_col_nmajor_);
  conv_im2col->row_major_to_col_nmajor_((float *)origin_weight, (float *)conv->packed_weight_, conv->output_c_,
                                        conv->input_c_ * conv->kernel_h_ * conv->kernel_w_);
}

void ConvIm2ColBaseInitGlobalVariable(ConvolutionBaseStruct *conv) {
  ConvolutionIm2ColBaseStruct *conv_im2col = (ConvolutionIm2ColBaseStruct *)conv;
  conv_im2col->oc_tile_ = C8NUM;
  conv_im2col->row_tile_ = C12NUM;
  conv_im2col->row_major_to_col_nmajor_ = RowMajor2Col8Major;
}

int convolution_im2col_base_release(KernelBase *self) {
  ConvolutionBaseStruct *conv = (ConvolutionBaseStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv);
  ConvBaseRelease(conv);
  return NNACL_OK;
}

int convolution_im2col_base_compute(KernelBase *self) {
  ConvolutionIm2ColBaseStruct *conv_im2col = (ConvolutionIm2ColBaseStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_im2col);

  int ret = conv_im2col->init_tmp_buffer_(conv_im2col);
  if (ret != NNACL_OK) {
    ConvIm2ColBaseFreeTmpBuffer(conv_im2col);
    return ret;
  }

  float *output_addr = self->out_[OUTPUT_INDEX]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(output_addr);
  if (!conv_im2col->output_need_align_) {
    conv_im2col->tmp_output_ = output_addr;
  }

  ret = ConvBaseRepackWeight(&conv_im2col->conv_);
  if (ret != NNACL_OK) {
    ConvIm2ColBaseFreeTmpBuffer(conv_im2col);
    return ret;
  }

  ret = self->env_->parallel_launch(self->env_->thread_pool_, ConvIm2ColBaseImpl, self, self->thread_nr_);
  ConvIm2ColBaseFreeTmpBuffer(conv_im2col);
  return ret;
}

int convolution_im2col_base_resize(KernelBase *self) {
  ConvolutionBaseStruct *conv = (ConvolutionBaseStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv);

  int ret = ConvBaseCheckResizeValid(conv);
  if (ret != NNACL_OK) {
    return ret;
  }

  ret = ConvBasePrepare(conv);
  if (ret != NNACL_OK) {
    return ret;
  }

  return ConvIm2ColBaseUpdateThreadNumProcess(self, TC_PTYPE(PrimType_Conv2DFusion), 0, 0, 0);
}

int convolution_im2col_base_prepare(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < TWO_TENSOR, NNACL_INPUT_TENSOR_ERROR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_OUTPUT_TENSOR_ERROR);

  ConvolutionIm2ColBaseStruct *conv_im2col = (ConvolutionIm2ColBaseStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_im2col);

  conv_im2col->conv_.init_global_variable_(&conv_im2col->conv_);

  if (self->train_session_) {
    int kernel_hw = conv_im2col->conv_.kernel_h_ * conv_im2col->conv_.kernel_w_;
    int oc_block_num = UP_ROUND(conv_im2col->conv_.output_c_, conv_im2col->oc_tile_);

    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv_im2col->conv_.input_c_, kernel_hw, NNACL_ERR);
    int kernel_chw = conv_im2col->conv_.input_c_ * kernel_hw;
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(oc_block_num, kernel_chw, NNACL_ERR);
    int pack_weight_size = oc_block_num * kernel_chw;

    self->work_size_ = pack_weight_size * sizeof(float);
  }

  return ConvBaseInitConvWeightBias(&conv_im2col->conv_);
}

ConvolutionBaseStruct *CreateConvIm2ColBase(ConvParameter *conv_param) {
  ConvolutionIm2ColBaseStruct *conv_im2col = (ConvolutionIm2ColBaseStruct *)malloc(sizeof(ConvolutionIm2ColBaseStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(conv_im2col);
  memset(conv_im2col, 0, sizeof(ConvolutionIm2ColBaseStruct));

  conv_im2col->init_tmp_buffer_ = ConvIm2ColBaseInitTmpBuffer;

  conv_im2col->conv_.malloc_weight_bias_ = ConvIm2ColBaseMallocWeightBiasData;
  conv_im2col->conv_.run_impl_ = ConvIm2ColBaseRunImpl;
  conv_im2col->conv_.pack_weight_ = ConvIm2ColBasePackWeight;
  conv_im2col->conv_.init_global_variable_ = ConvIm2ColBaseInitGlobalVariable;

  conv_im2col->conv_.base_.compute = convolution_im2col_base_compute;
  conv_im2col->conv_.base_.prepare = convolution_im2col_base_prepare;
  conv_im2col->conv_.base_.resize = convolution_im2col_base_resize;
  conv_im2col->conv_.base_.release = convolution_im2col_base_release;

  return (ConvolutionBaseStruct *)conv_im2col;
}
