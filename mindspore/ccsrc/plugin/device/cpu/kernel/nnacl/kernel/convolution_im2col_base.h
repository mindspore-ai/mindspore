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

#ifndef NNACL_KERNEL_CONVOLLUTION_IM2COL_BASE_H_
#define NNACL_KERNEL_CONVOLLUTION_IM2COL_BASE_H_

#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"
#include "nnacl/kernel.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/kernel/convolution_base.h"

typedef struct ConvolutionIm2ColBaseStruct {
  ConvolutionBaseStruct conv_;
  int oc_tile_;
  int row_tile_;

  float *tmp_output_;
  float *packed_input_;
  float *col_major_input_;
  bool output_need_align_;

  void (*row_major_to_col_nmajor_)(const float *src_ptr, float *dst_ptr, int row, int col);
  int (*init_tmp_buffer_)(struct ConvolutionIm2ColBaseStruct *conv_im2col);
} ConvolutionIm2ColBaseStruct;

int ConvIm2ColBaseMallocWeightBiasData(ConvolutionBaseStruct *conv);
int ConvIm2ColBaseInitTmpBuffer(ConvolutionIm2ColBaseStruct *conv_im2col);
int ConvIm2ColBaseImpl(void *cdata, int task_id, float l, float r);
void ConvIm2ColBaseFreeTmpBuffer(ConvolutionIm2ColBaseStruct *conv_im2col);
void ConvIm2ColBasePackWeight(ConvolutionBaseStruct *conv);
int ConvIm2ColBaseRunImpl(ConvolutionBaseStruct *conv, int task_id);
int ConvolutionIm2colBaseCompute(KernelBase *self);
int ConvolutionIm2colBasePrepare(KernelBase *self);
int ConvolutionIm2colBaseResize(KernelBase *self);
int ConvolutionIm2colBaseRelease(KernelBase *self);
ConvolutionBaseStruct *CreateConvIm2ColBase(ConvParameter *conv_param);

#endif  // NNACL_KERNEL_CONVOLLUTION_IM2COL_BASE_H_
