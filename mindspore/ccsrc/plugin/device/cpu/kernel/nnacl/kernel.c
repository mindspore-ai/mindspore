/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "nnacl/kernel.h"
#include "nnacl/tensor_c.h"
#include "nnacl/op_base.h"
static KernelCreator g_kernelCreatorRegistry[PrimType_MAX][kDataTypeMax][NUM_OF_FORMAT];

void RegKernelCreator(int opType, LiteDataType dataType, TensorCFormat format, KernelCreator creator) {
  g_kernelCreatorRegistry[opType][dataType][format] = creator;
}

KernelStru *CreateKernel(OpParameter *param, TensorC *in[], size_t insize, TensorC *out[], size_t outsize) {
  int format = in[kInputIndex]->format_;
  int dtype = in[kInputIndex]->data_type_;
  return g_kernelCreatorRegistry[param->type_][format][dtype](param, in, insize, out, outsize);
}
