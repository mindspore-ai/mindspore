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

#ifndef NNACL_KERNEL_CONVOLLUTION_1X1_H_
#define NNACL_KERNEL_CONVOLLUTION_1X1_H_

#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"
#include "nnacl/kernel.h"
#include "nnacl/kernel/convolution_base.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/matmul_parameter.h"

typedef struct Convolution1x1Struct {
  ConvolutionBaseStruct conv_;
  MatMulParameter matmul_param_;
  int row_tile_;
  int col_tile_;
  bool pre_trans_input_;
  float *input_ptr_;
  float *output_ptr_;
  float *pack_input_;
  bool multi_thread_by_hw_;
  int thread_stride_;
} Convolution1x1Struct;

ConvolutionBaseStruct *CreateConvolution1x1(ConvParameter *conv_param);

#endif  // NNACL_KERNEL_CONVOLLUTION_1X1_H_
