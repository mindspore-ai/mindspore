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
#include "nnacl/experimental/ms_core.h"
#ifdef _MSC_VER
#include "nnacl/experimental/conv.h"
#include "nnacl/kernel/exp.h"
#endif

static KernelCreator g_kernelCreatorRegistry[PrimType_MAX][Format_MAX][16];
#define REGIST_DT(DT) (DT - kNumberTypeBegin - 1)

void RegKernelCreator(int opType, int format, int dataType, KernelCreator creator) {
  g_kernelCreatorRegistry[opType][format][REGIST_DT(dataType)] = creator;
}

void Init_MSC_VER_kernels(void) {
#ifdef _MSC_VER
  /* VS env do not support automatic register
   * register here first time */
  static bool inited = false;
  if (inited == false) {
    g_kernelCreatorRegistry[PrimType_Conv2DFusion][Format_NC4HW4][REGIST_DT(kNumberTypeFloat32)] = CreateConv;
    g_kernelCreatorRegistry[PrimType_ExpFusion][Format_NHWC][REGIST_DT(kNumberTypeFloat32)] = CreateExp;
    g_kernelCreatorRegistry[PrimType_ExpFusion][Format_NHWC][REGIST_DT(kNumberTypeFloat16)] = CreateExp;
    g_kernelCreatorRegistry[PrimType_ExpFusion][Format_NCHW][REGIST_DT(kNumberTypeFloat32)] = CreateExp;
    g_kernelCreatorRegistry[PrimType_ExpFusion][Format_NCHW][REGIST_DT(kNumberTypeFloat16)] = CreateExp;
    g_kernelCreatorRegistry[PrimType_ExpFusion][Format_NC4HW4][REGIST_DT(kNumberTypeFloat32)] = CreateExp;
    g_kernelCreatorRegistry[PrimType_ExpFusion][Format_NC8HW8][REGIST_DT(kNumberTypeFloat16)] = CreateExp;
    inited = true;
  }
#endif
  return;
}

bool SupportKernelC(int opType, int format, int dataType) {
  Init_MSC_VER_kernels();
  const int length = 16;
  if (REGIST_DT(dataType) < 0 || REGIST_DT(dataType) >= length) {
    return false;
  }
  KernelCreator creator = g_kernelCreatorRegistry[opType][format][REGIST_DT(dataType)];
  return creator != NULL;
}

KernelBase *CreateKernel(OpParameter *param, TensorC *in, size_t insize, TensorC *out, size_t outsize, int data_type,
                         FormatC format, ExecEnv *env) {
  Init_MSC_VER_kernels();
  KernelCreator creator = g_kernelCreatorRegistry[param->type_][format][REGIST_DT(data_type)];
  if (creator == NULL) {
    return NULL;
  }
  KernelBase *kernel_base = creator(param, data_type, format);
  kernel_base->env = env;
  kernel_base->param = param;
  kernel_base->in = in;
  kernel_base->insize = insize;
  kernel_base->out = out;
  kernel_base->outsize = outsize;
  return kernel_base;
}

CoreFuncs *GetCoreFuncs(bool use_fp16) {
  static CoreFuncs core;
  InitCore(&core);

#ifdef ENABLE_AVX512
  static CoreFuncs core_avx512;
  InitCore(&core_avx512);
  InitSseCore(&core_avx512);
  InitAvxCore(&core_avx512);
  InitAvx512Core(&core_avx512);
  return &core_avx512;
#endif

#ifdef ENABLE_AVX
  static CoreFuncs core_avx;
  InitCore(&core_avx);
  InitSseCore(&core_avx);
  InitAvxCore(&core_avx);
  return &core_avx;
#endif

#ifdef ENABLE_SSE
  static CoreFuncs core_sse;
  InitCore(&core_sse);
  InitSseCore(&core_sse);
  return &core_sse;
#endif

#ifdef ENABLE_ARM32
  static CoreFuncs core_arm32;
  InitCore(&core_arm32);
  InitArm32Core(&core_arm32);
  return &core_arm32;
#endif

#ifdef ENABLE_ARM64
  static CoreFuncs core_fp32;
  InitCore(&core_fp32);
  InitFp32Core(&core_fp32);
  static CoreFuncs core_fp16;
  InitCore(&core_fp16);
#ifdef ENABLE_FP16
  InitFp16Core(&core_fp16);
#endif
  return use_fp16 ? &core_fp16 : &core_fp32;
#endif

  return &core;
}
