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
#include "nnacl/experimental/fp32_funcs.h"
#include "nnacl/experimental/fp16_funcs.h"

static KernelCreator g_kernelCreatorRegistry[PrimType_MAX][16];

void RegKernelCreator(int opType, int dataType, KernelCreator creator) {
  g_kernelCreatorRegistry[opType][dataType - kNumberTypeBegin - 1] = creator;
}

KernelBase *CreateKernel(OpParameter *param, TensorC *in[], size_t insize, TensorC *out[], size_t outsize) {
  int dtype = in[kInputIndex]->data_type_;
  KernelCreator creator = g_kernelCreatorRegistry[param->type_][dtype - kNumberTypeBegin - 1];
  if (creator == NULL) {
    return NULL;
  }
  return creator(param, in, insize, out, outsize);
}

ExecEnv *GetExecEnv() {
  static ExecEnv kc;
  return &kc;
}

CoreFuncs *GetCoreFuncs(bool use_fp16) {
  static CoreFuncs fp23funcs;
  InitFp32Funcs(&fp23funcs);
  static CoreFuncs fp16funcs;
  InitFp16Funcs(&fp16funcs);

  if (use_fp16) {
    return &fp16funcs;
  }

  return &fp23funcs;
}
