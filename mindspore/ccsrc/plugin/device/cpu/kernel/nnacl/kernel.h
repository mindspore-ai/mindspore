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
#include "nnacl/tensor_c.h"

typedef struct KernelContext {
  void *(*alloc)(size_t sz);
  void (*free)(void *ptr);
  int threadNum;
  void (*parallelLaunch)(void *task, void *param, int taskNr);
} KernelContext;

typedef struct KernelStru {
  int (*init)(struct KernelStru *self, KernelContext *ctx);
  int (*release)(struct KernelStru *self);
  int (*compute)(struct KernelStru *self);
  int (*infershape)(OpParameter *param, TensorC *in[], size_t insize, TensorC *out[], size_t outsize);
  OpParameter *param;
  TensorC **in;  // in/out tensor's space should be managed by the invoker
  size_t insize;
  TensorC **out;
  size_t outsize;
  KernelContext *ctx;
  void *buf[4];
} KernelStru;

KernelStru *CreateKernel(OpParameter *param, TensorC *in[], size_t insize, TensorC *out[], size_t outsize);
typedef KernelStru *(*KernelCreator)(OpParameter *param, TensorC *in[], size_t insize, TensorC *out[], size_t outsize);
void RegKernelCreator(int opType, LiteDataType dataType, TensorCFormat format, KernelCreator func);

#ifdef _MSC_VER
#define REG_KERNEL_CREATOR(op, op_type, data_type, format, func)
#else
#define REG_KERNEL_CREATOR(op, op_type, data_type, format, func) \
  __attribute__((constructor(102))) void Reg##op##Creator() { RegKernelCreator(op_type, data_type, format, func); }
#endif

#endif
