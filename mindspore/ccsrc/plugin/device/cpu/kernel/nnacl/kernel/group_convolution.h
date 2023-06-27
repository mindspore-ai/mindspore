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

#ifndef NNACL_KERNEL_GROUP_CONVOLUTION_H_
#define NNACL_KERNEL_GROUP_CONVOLUTION_H_

#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"
#include "nnacl/kernel.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/kernel/convolution_base.h"

typedef struct GroupConvolutionStruct {
  ConvolutionBaseStruct conv_base_;
  KernelBase **group_convs_;
  ConvParameter new_conv_param_;
  TypeIdC data_type_;
  int group_;

  void *origin_input_data_;
  void *origin_output_data_;

  float *sub_in_src_;
  float *sub_in_dst_;
  float *sub_out_src_;
  float *sub_out_dst_;

  int sub_in_c_;
  int ori_in_c_;
  int sub_out_c_;
  int ori_out_c_;
} GroupConvolutionStruct;

KernelBase *CreateGroupConvolution(ConvParameter *conv_param, TypeIdC data_type);

#endif  // NNACL_KERNEL_GROUP_CONVOLUTION_H_
