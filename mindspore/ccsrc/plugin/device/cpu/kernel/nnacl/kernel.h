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
#ifndef MINDSPORE_NNACL_KERNEL_H_
#define MINDSPORE_NNACL_KERNEL_H_

#include "nnacl/op_base.h"
#include "nnacl/infer/common_infer.h"
#include "nnacl/experimental/ms_core.h"

typedef struct ExecEnv {
  void *allocator;
  void *threadPool;
  void *(*alloc)(void *allocator, size_t sz);
  void (*free)(void *allocator, void *ptr);
  int threadNum;
  int (*parallelLaunch)(void *threadPool, void *task, void *param, int taskNr);
} ExecEnv;

typedef struct KernelBase {
  int (*prepare)(struct KernelBase *self);  // prepare, e.g. pack weight
  int (*release)(struct KernelBase *self);
  int (*compute)(struct KernelBase *self);
  int (*inferShape)(struct KernelBase *self);
  int (*resize)(struct KernelBase *self);
  OpParameter *param;
  // by design, kernelBase's methods are not responsible for input/output tensors' management, user must be invokes
  // KernelBase's infer shape and allocate/free input/output tensor at necessary time.
  TensorC *in;
  size_t insize;
  TensorC *out;
  size_t outsize;
  ExecEnv *env;
  bool inferShape_;
  CoreFuncs *funcs;
} KernelBase;

#ifdef _MSC_VER
#define REG_KERNEL_CREATOR(op_type, format, data_type, func)
#else
#define REG_KERNEL_CREATOR(op, format, data_type, func)                          \
  __attribute__((constructor(102))) void Reg##op##format##data_type##Creator() { \
    RegKernelCreator(op, format, data_type, func);                               \
  }
#endif

typedef KernelBase *(*KernelCreator)(OpParameter *param, int data_type, FormatC format);
void RegKernelCreator(int opType, int format, int dataType, KernelCreator func);
CoreFuncs *GetCoreFuncs(bool use_fp16);

#ifdef __cplusplus
extern "C" {
#endif
KernelBase *CreateKernel(OpParameter *param, TensorC *in, size_t insize, TensorC *out, size_t outsize, int data_type,
                         FormatC format, ExecEnv *env);
bool SupportKernelC(int opType, int format, int dataType);
#ifdef __cplusplus
}
#endif
#endif
