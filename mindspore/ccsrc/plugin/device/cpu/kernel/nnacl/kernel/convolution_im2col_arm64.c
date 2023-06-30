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

#ifdef ENABLE_ARM64
#include "nnacl/kernel/convolution_im2col_arm64.h"
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/fp32/conv_common_fp32.h"

void ConvIm2ColARM64InitGlobalVariable(ConvolutionBaseStruct *conv) {
  ConvolutionIm2ColBaseStruct *conv_im2col = (ConvolutionIm2ColBaseStruct *)conv;
  conv_im2col->oc_tile_ = C8NUM;
  conv_im2col->row_tile_ = C12NUM;
  conv_im2col->row_major_to_col_nmajor_ = RowMajor2Col8Major;
}

int ConvIm2ColARM64RunImpl(struct ConvolutionBaseStruct *conv, int task_id) {
  ConvolutionIm2ColBaseStruct *conv_im2col = (ConvolutionIm2ColBaseStruct *)conv;
  NNACL_CHECK_NULL_RETURN_ERR(conv_im2col);
  float *ori_input_data = (float *)conv->base_.in_[FIRST_INPUT]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(ori_input_data);
  ConvParameter *conv_param = (ConvParameter *)conv->base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);

  if (conv->out_format_ != Format_NC4HW4) {
    if (conv->use_batch_cut_flag_) {
      ConvFp32CutByBatch(ori_input_data, conv_im2col->packed_input_, (float *)conv->packed_weight_,
                         (float *)conv->bias_data_, conv_im2col->col_major_input_, conv_im2col->tmp_output_, task_id,
                         conv_param);
    } else {
      ConvFp32(ori_input_data, conv_im2col->packed_input_, (float *)conv->packed_weight_, (float *)conv->bias_data_,
               conv_im2col->col_major_input_, conv_im2col->tmp_output_, task_id, conv_param);
    }
  } else {
    ConvFp32OutNC4HW4(ori_input_data, conv_im2col->packed_input_, (float *)conv->packed_weight_,
                      (float *)conv->bias_data_, conv_im2col->col_major_input_, conv_im2col->tmp_output_, task_id,
                      conv_param);
  }
  return NNACL_OK;
}

ConvolutionBaseStruct *CreateConvIm2ColARM64(ConvParameter *conv_param) {
  ConvolutionIm2ColBaseStruct *conv_im2col = (ConvolutionIm2ColBaseStruct *)malloc(sizeof(ConvolutionIm2ColBaseStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(conv_im2col);
  memset(conv_im2col, 0, sizeof(ConvolutionIm2ColBaseStruct));

  conv_im2col->init_tmp_buffer_ = ConvIm2ColBaseInitTmpBuffer;
  conv_im2col->conv_.malloc_weight_bias_ = ConvIm2ColBaseMallocWeightBiasData;
  conv_im2col->conv_.init_global_variable_ = ConvIm2ColARM64InitGlobalVariable;
  conv_im2col->conv_.run_impl_ = ConvIm2ColARM64RunImpl;
  conv_im2col->conv_.pack_weight_ = ConvIm2ColBasePackWeight;

  conv_im2col->conv_.base_.Compute = ConvolutionIm2colBaseCompute;
  conv_im2col->conv_.base_.Prepare = ConvolutionIm2colBasePrepare;
  conv_im2col->conv_.base_.Resize = ConvolutionIm2colBaseResize;
  conv_im2col->conv_.base_.Release = ConvolutionIm2colBaseRelease;

  return (ConvolutionBaseStruct *)conv_im2col;
}
#endif
