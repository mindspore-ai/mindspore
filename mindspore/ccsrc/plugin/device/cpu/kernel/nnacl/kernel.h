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

typedef struct ExecEnv {
  void *(*alloc)(size_t sz);
  void (*free)(void *ptr);
  int threadNum;
  void (*parallelLaunch)(void *task, void *param, int taskNr);
} ExecEnv;

typedef struct KernelBase {
  int (*prepare)(struct KernelBase *self, ExecEnv *env);  // prepare, e.g. pack weight
  int (*release)(struct KernelBase *self);
  int (*compute)(struct KernelBase *self);
  int (*inferShape)(struct KernelBase *self);
  int (*resize)(struct KernelBase *self, TensorC *in[], size_t insize, TensorC *out[], size_t outsize);
  OpParameter *param;
  // by design, kernelBase's methods are not responsible for input/output tensors' management, user should invokes
  // KernelBase's infer shape and allocate/free input/output tensor at necessary time.
  TensorC **in;
  size_t insize;
  TensorC **out;
  size_t outsize;
  ExecEnv *env;
  bool inferShape_;
} KernelBase;

KernelBase *CreateKernel(OpParameter *param, TensorC *in[], size_t insize, TensorC *out[], size_t outsize);
typedef KernelBase *(*KernelCreator)(OpParameter *param, TensorC *in[], size_t insize, TensorC *out[], size_t outsize);
void RegKernelCreator(int opType, int dataType, KernelCreator func);

#ifdef _MSC_VER
#define REG_KERNEL_CREATOR(op, op_type, data_type, func)
#else
#define REG_KERNEL_CREATOR(op, op_type, data_type, func) \
  __attribute__((constructor(102))) void Reg##op##Creator() { RegKernelCreator(op_type, data_type, func); }
#endif

#endif
