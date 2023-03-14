/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "nnacl/kernel/convolution.h"
#include "nnacl/kernel/convolution_1x1.h"
#include "nnacl/tensor_c.h"
#include "nnacl/op_base.h"
#include "nnacl/conv_parameter.h"

KernelBase *CreateConvolution(OpParameter *param, int data_type) {
  ConvParameter *conv = (ConvParameter *)param;
  if (conv->kernel_h_ == 1 && conv->kernel_w_ == 1) {
    return CreateConv1x1(param, data_type);
  }
  return NULL;
}

REG_KERNEL_CREATOR(PrimType_Conv2DFusion, kNumberTypeFloat32, CreateConvolution);
REG_KERNEL_CREATOR(PrimType_Conv2DFusion, kNumberTypeFloat16, CreateConvolution);
