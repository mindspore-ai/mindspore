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
#include "nnacl/kernel/init_exec_env.h"

static KernelCreator g_kernelCreatorRegistry[PrimType_MAX][16];

void RegKernelCreator(int opType, int dataType, KernelCreator creator) {
  g_kernelCreatorRegistry[opType][REGIST_DT(dataType)] = creator;
}

void Init_MSC_VER_kernels(void) {
#ifdef _MSC_VER
  /* VS env do not support automatic register
   * register here first time */
  static bool inited = false;
  if (inited == false) {
    init_vs_kernels(g_kernelCreatorRegistry);
    inited = true;
  }
#endif
  return;
}

bool checkOpValid(int opType) {
  if (opType < PrimType_MIN || opType >= PrimType_MAX) {
    return false;
  }
  return true;
}

bool SupportKernelC(int opType, int dataType) {
  Init_MSC_VER_kernels();
  const int length = 16;
  if (REGIST_DT(dataType) < 0 || REGIST_DT(dataType) >= length) {
    return false;
  }
  if (!checkOpValid(opType)) {
    return false;
  }
  KernelCreator creator = g_kernelCreatorRegistry[opType][REGIST_DT(dataType)];
  return creator != NULL;
}

int DefaultThreadUpdate(int32_t type, int64_t load, int64_t store, int64_t unit, int thread) {
  return thread > 0 ? thread : 1;
}

int NNACLKernelInferShape(struct KernelBase *self) { return NNACL_ERR; }

int NNACLCheckKernelBase(KernelBase *kernel_base) {
  CheckExecEnv(kernel_base);

  if (kernel_base->param_ == NULL) {
    return NNACL_ERR;
  }

  if (kernel_base->thread_nr_ <= 0 || kernel_base->thread_nr_ > MAX_THREAD_NUM) {
    return NNACL_ERR;
  }

  if (kernel_base->in_size_ == 0 || kernel_base->in_ == NULL) {
    return NNACL_ERR;
  }
  if (kernel_base->out_size_ == 0 || kernel_base->out_ == NULL) {
    return NNACL_ERR;
  }
  return NNACL_OK;
}

KernelBase *CreateKernel(OpParameter *param, TensorC **ins, size_t in_size, TensorC **outs, size_t out_size,
                         int data_type, ExecEnv *env) {
  Init_MSC_VER_kernels();
  if (param == NULL) {
    return NULL;
  }
  if (!checkOpValid(param->type_)) {
    return NULL;
  }

  KernelCreator creator = g_kernelCreatorRegistry[param->type_][REGIST_DT(data_type)];
  if (creator == NULL) {
    return NULL;
  }

  KernelBase *kernel_base = creator(param, data_type);
  if (kernel_base == NULL) {
    return NULL;
  }

  kernel_base->InferShape = NNACLKernelInferShape;
  kernel_base->UpdateThread = DefaultThreadUpdate;
  kernel_base->env_ = env;
  kernel_base->param_ = param;
  kernel_base->thread_nr_ = param->thread_num_;
  kernel_base->train_session_ = param->is_train_session_;
  kernel_base->in_ = ins;
  kernel_base->in_size_ = in_size;
  kernel_base->out_ = outs;
  kernel_base->out_size_ = out_size;

  int ret = NNACLCheckKernelBase(kernel_base);
  if (ret != NNACL_OK) {
    return NULL;
  }

  return kernel_base;
}
