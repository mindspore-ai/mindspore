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

#ifndef NNACL_KERNEL_CONVOLLUTION_DEPTHWISE_SW_AVX_H_
#define NNACL_KERNEL_CONVOLLUTION_DEPTHWISE_SW_AVX_H_

#ifdef ENABLE_AVX
#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"
#include "nnacl/kernel.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/kernel/convolution_base.h"

typedef struct ConvolutionDepthwiseSWAVXStruct {
  ConvolutionBaseStruct conv_;
  SlidingWindowParam sliding_param_;
  int oc_tile_;
  float *packed_input_;
  float *packed_output_;
  bool input_need_align_;
  bool output_need_align_;
} ConvolutionDepthwiseSWAVXStruct;

KernelBase *CreateConvDwSWAVX(ConvParameter *conv_param);
#endif

#endif  // NNACL_KERNEL_CONVOLLUTION_DEPTHWISE_SW_AVX_H_
