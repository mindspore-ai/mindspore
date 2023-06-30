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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NNACL_KERNEL_CONVOLLUTION_WINOGRAD_BASE_H_
#define NNACL_KERNEL_CONVOLLUTION_WINOGRAD_BASE_H_

#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"
#include "nnacl/kernel.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/kernel/convolution_base.h"
#include "nnacl/fp32/winograd_utils.h"

#define CONVOLUTION_WINOGRAD_MATRIX_SIZE 64
#define CONVOLUTION_WINOGRAD_TMP_BUFFER_SIZE 5
#define CONVOLUTION_WINOGRAD_INPUT_UNIT_SIZE 8

typedef float *TmpBufferAddress;

typedef struct ConvolutionWinogradBaseStruct {
  ConvolutionBaseStruct conv_;

  int kernel_unit_;
  int input_unit_;
  int output_unit_;
  int oc_block_;
  int tile_num_;
  int tmp_data_tile_;
  float *tmp_data_;
  float *trans_input_;
  float *gemm_out_;
  float *col_buffer_;
  float *opt_input_trans_;
  float matrix_g_[CONVOLUTION_WINOGRAD_MATRIX_SIZE];
  float matrix_gt_[CONVOLUTION_WINOGRAD_MATRIX_SIZE];
  TmpBufferAddress tmp_buffer_address_list_[CONVOLUTION_WINOGRAD_TMP_BUFFER_SIZE];
  TransFuncList transfer_functions_;

  int (*config_input_output_)(struct ConvolutionWinogradBaseStruct *winograd);
} ConvolutionWinogradBaseStruct;

void ConvWinoBasePackWeight(ConvolutionBaseStruct *conv);
int ConvWinoBaseConfigInputOutput(ConvolutionWinogradBaseStruct *winograd);
int ConvWinoBaseRunImpl(ConvolutionBaseStruct *conv, int task_id);
int ConvWinoBaseMallocWeightBiasData(ConvolutionBaseStruct *conv);
int ConvolutionWinogradBasePrepare(KernelBase *self);
int ConvolutionWinogradBaseResize(KernelBase *self);
int ConvolutionWinogradBaseRelease(KernelBase *self);
int ConvolutionWinogradBaseCompute(KernelBase *self);
ConvolutionWinogradBaseStruct *CreateConvWinogradBase(ConvParameter *conv_param);

#endif  // NNACL_KERNEL_CONVOLLUTION_WINOGRAD_BASE_H_
