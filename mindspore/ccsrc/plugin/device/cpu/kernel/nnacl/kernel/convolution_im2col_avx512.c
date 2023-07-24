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

#ifdef ENABLE_AVX512
#include "nnacl/kernel/convolution_im2col_avx512.h"
#include "nnacl/fp32/conv_im2col_avx512_fp32.h"
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/tensor_c.h"

void ConvIm2ColAVX512InitGlobalVariable(ConvolutionBaseStruct *conv) {
  ConvolutionIm2ColBaseStruct *conv_im2col = (ConvolutionIm2ColBaseStruct *)conv;
  conv_im2col->oc_tile_ = C16NUM;
  conv_im2col->row_tile_ =
    MSMIN(UP_DIV(conv_im2col->conv_.compute_.out_hw_, conv_im2col->conv_.base_.thread_nr_), C150NUM);
  conv_im2col->row_major_to_col_nmajor_ = RowMajor2Col64Major;
}

int ConvIm2ColAVX512InitTmpBuffer(struct ConvolutionIm2ColBaseStruct *conv_im2col) {
  ExecEnv *env = conv_im2col->conv_.base_.env_;
  NNACL_CHECK_NULL_RETURN_ERR(env);
  ConvComputeParam *compute = &conv_im2col->conv_.compute_;
  NNACL_CHECK_NULL_RETURN_ERR(compute);

  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(compute->kernel_hw_, compute->in_c_, NNACL_ERR);
  int kernel_chw = compute->kernel_hw_ * compute->in_c_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(kernel_chw, conv_im2col->conv_.base_.thread_nr_, NNACL_ERR);
  int total_kernel_chw = kernel_chw * conv_im2col->conv_.base_.thread_nr_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(total_kernel_chw, conv_im2col->row_tile_, NNACL_ERR);
  size_t unit_size = total_kernel_chw * conv_im2col->row_tile_;

  if (conv_im2col->packed_input_ != NULL) {
    env->Free(env->allocator_, conv_im2col->packed_input_);
    conv_im2col->packed_input_ = NULL;
  }
  conv_im2col->packed_input_ = env->Alloc(env->allocator_, unit_size * sizeof(float));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv_im2col->packed_input_);

  conv_im2col->output_need_align_ = compute->out_c_ % conv_im2col->oc_tile_ != 0;
  if (conv_im2col->output_need_align_) {
    if (conv_im2col->tmp_output_ != NULL) {
      env->Free(env->allocator_, conv_im2col->tmp_output_);
      conv_im2col->tmp_output_ = NULL;
    }

    // avx512 need to malloc dst aligned to C16NUM
    int oc_algin = UP_ROUND(compute->out_c_, conv_im2col->oc_tile_);
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(compute->out_n_, compute->out_hw_, NNACL_ERR);
    int output_bhw = compute->out_n_ * compute->out_hw_;
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(output_bhw, oc_algin, NNACL_ERR);
    size_t pack_output_size = output_bhw * compute->out_w_ * oc_algin;

    conv_im2col->tmp_output_ = env->Alloc(env->allocator_, pack_output_size * sizeof(float));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv_im2col->tmp_output_);
  }

  return NNACL_OK;
}

int ConvIm2ColAVX512RunImpl(struct ConvolutionBaseStruct *conv, int task_id) {
  if (conv->out_format_ == Format_NC4HW4) {
    return NNACL_CONVOLUTION_AVX512_UNSUPPORT_FORMAT;
  }

  ConvolutionIm2ColBaseStruct *conv_im2col = (ConvolutionIm2ColBaseStruct *)conv;
  NNACL_CHECK_NULL_RETURN_ERR(conv_im2col);
  ConvParameter *conv_param = (ConvParameter *)conv->base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);
  float *ori_input_data = (float *)conv->base_.in_[FIRST_INPUT]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(ori_input_data);

  if (conv->use_batch_cut_flag_) {
    ConvIm2ColAVX512Fp32CutByBatch(ori_input_data, conv_im2col->packed_input_, (float *)conv->packed_weight_,
                                   (float *)conv->bias_data_, conv_im2col->tmp_output_, task_id, conv_param,
                                   conv_im2col->row_tile_);
  } else {
    ConvIm2ColAVX512Fp32(ori_input_data, conv_im2col->packed_input_, (float *)conv->packed_weight_,
                         (float *)conv->bias_data_, conv_im2col->tmp_output_, task_id, conv_param,
                         conv_im2col->row_tile_);
  }
  return NNACL_OK;
}

int ConvolutionIm2colAvx512Compute(KernelBase *self) {
  ConvolutionIm2ColBaseStruct *conv_im2col = (ConvolutionIm2ColBaseStruct *)self;
  int ret = conv_im2col->init_tmp_buffer_(conv_im2col);
  if (ret != NNACL_OK) {
    ConvIm2ColBaseFreeTmpBuffer(conv_im2col);
    return ret;
  }

  float *output_addr = (float *)self->out_[OUTPUT_INDEX]->data_;
  if (!conv_im2col->output_need_align_) {
    conv_im2col->tmp_output_ = output_addr;
  }

  ret = ConvBaseRepackWeight(&conv_im2col->conv_);
  if (ret != NNACL_OK) {
    ConvIm2ColBaseFreeTmpBuffer(conv_im2col);
    return ret;
  }

  ret = self->env_->ParallelLaunch(self->env_->thread_pool_, ConvIm2ColBaseImpl, self, self->thread_nr_);

  if (conv_im2col->output_need_align_) {
    PackNHWCXToNHWCFp32(conv_im2col->tmp_output_, output_addr, conv_im2col->conv_.compute_.out_n_,
                        conv_im2col->conv_.compute_.out_hw_, conv_im2col->conv_.compute_.out_c_, conv_im2col->oc_tile_);
  } else {
    conv_im2col->tmp_output_ = NULL;
  }

  ConvIm2ColBaseFreeTmpBuffer(conv_im2col);
  return ret;
}

ConvolutionBaseStruct *CreateConvIm2ColAVX512(ConvParameter *conv_param) {
  ConvolutionIm2ColBaseStruct *conv_im2col = (ConvolutionIm2ColBaseStruct *)malloc(sizeof(ConvolutionIm2ColBaseStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(conv_im2col);
  memset(conv_im2col, 0, sizeof(ConvolutionIm2ColBaseStruct));

  conv_im2col->init_tmp_buffer_ = ConvIm2ColAVX512InitTmpBuffer;
  conv_im2col->conv_.malloc_weight_bias_ = ConvIm2ColBaseMallocWeightBiasData;
  conv_im2col->conv_.init_global_variable_ = ConvIm2ColAVX512InitGlobalVariable;
  conv_im2col->conv_.run_impl_ = ConvIm2ColAVX512RunImpl;
  conv_im2col->conv_.pack_weight_ = ConvIm2ColBasePackWeight;

  conv_im2col->conv_.base_.Compute = ConvolutionIm2colAvx512Compute;
  conv_im2col->conv_.base_.Prepare = ConvolutionIm2colBasePrepare;
  conv_im2col->conv_.base_.Resize = ConvolutionIm2colBaseResize;
  conv_im2col->conv_.base_.Release = ConvolutionIm2colBaseRelease;

  return (ConvolutionBaseStruct *)conv_im2col;
}
#endif
