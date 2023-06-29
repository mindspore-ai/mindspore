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

#ifndef NNACL_KERNEL_CONVOLLUTION_SLIDEWINDOW_H_
#define NNACL_KERNEL_CONVOLLUTION_SLIDEWINDOW_H_

#if defined(ENABLE_AVX) || defined(ENABLE_ARM64)
#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"
#include "nnacl/kernel.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/kernel/convolution_base.h"
#include "nnacl/matmul_parameter.h"

typedef struct ConvolutionSWStruct {
  ConvolutionBaseStruct conv_;
  SlidingWindowParam sw_param_;
  int oc_tile_;
  int in_tile_;
  int oc_res_;
  int ic_res_;
  float *output_data_;
  float *input_data_;
} ConvolutionSWStruct;

int ConvolutionSWPrepare(KernelBase *self);
int ConvolutionSWCompute(KernelBase *self);
int ConvolutionSWResize(KernelBase *self);
int ConvolutionSWRelease(KernelBase *self);
void ConvSWPackWeight(ConvolutionBaseStruct *conv);
int ConvSWMallocWeightBiasData(ConvolutionBaseStruct *conv);
#endif
#endif  // NNACL_KERNEL_CONVOLLUTION_SLIDEWINDOW_H_
