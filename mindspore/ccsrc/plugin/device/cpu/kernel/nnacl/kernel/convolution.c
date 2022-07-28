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

KernelBase *CreateConvolution(OpParameter *param, TensorC *in, size_t insize, TensorC *out, size_t outsize,
                              int data_type, FormatC format) {
  TensorC weight = in[1];
  if (GetWidth(&weight) == 1 && GetHeight(&weight) == 1) {
    return CreateConv1x1(param, in, insize, out, outsize, data_type, format);
  }
  return NULL;
}

REG_KERNEL_CREATOR(PrimType_Conv2DFusion, Format_NC4HW4, kNumberTypeFloat32, CreateConvolution);
REG_KERNEL_CREATOR(PrimType_Conv2DFusion, Format_NC8HW8, kNumberTypeFloat16, CreateConvolution);
