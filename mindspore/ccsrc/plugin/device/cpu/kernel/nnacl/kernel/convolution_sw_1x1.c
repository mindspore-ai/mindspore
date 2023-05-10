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

#include "nnacl/kernel/convolution_sw_1x1.h"
ConvolutionBaseStruct *CreateConvolutionSW1x1(ConvParameter *conv_param) {
  MatMulParameter *matmul_param = (MatMulParameter *)malloc(sizeof(MatMulParameter));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(matmul_param);

  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv_param->output_h_, conv_param->output_w_, NULL);
  matmul_param->row_ = conv_param->output_h_ * conv_param->output_w_;
  matmul_param->col_ = conv_param->output_channel_;
  matmul_param->deep_ = conv_param->input_channel_;
  matmul_param->batch = conv_param->input_batch_;
  matmul_param->op_parameter_ = conv_param->op_parameter_;
  matmul_param->act_type_ = conv_param->act_type_;
  matmul_param->a_transpose_ = false;
  matmul_param->b_transpose_ = true;
  //  matmul_param->a_const_ = input_const_;
  //  matmul_param->b_const_ = weight_const_;

  ConvolutionSW1x1Struct *sw_1x1 = (ConvolutionSW1x1Struct *)malloc(sizeof(ConvolutionSW1x1Struct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(sw_1x1);
  memset(sw_1x1, 0, sizeof(ConvolutionSW1x1Struct));

  sw_1x1->matmul_param_ = matmul_param;
  return (ConvolutionBaseStruct *)sw_1x1;
}
