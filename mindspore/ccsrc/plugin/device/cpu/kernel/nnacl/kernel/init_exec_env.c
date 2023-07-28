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

#include "nnacl/kernel/init_exec_env.h"

#define NNACLMaxAllocSize (2000 * 1024 * 1024)
ExecEnv nnacl_default_env;

void *NNACLDefaultAlloc(void *allocator, size_t sz) {
  if (sz == 0 || sz > NNACLMaxAllocSize) {
    return NULL;
  }
  return malloc(sz);
}

void NNACLDefaultFree(void *allocator, void *ptr) { return free(ptr); }

int NNACLDefaultParallelLunch(void *threadPool, void *task, void *param, int taskNr) {
  int (*function)(void *cdata, int task_id, float l, float r) = task;
  int ret = 0;
  for (int i = 0; i < taskNr; i++) {
    ret += function(param, i, 0, 1);
  }
  return ret == NNACL_OK ? NNACL_OK : NNACL_ERR;
}

void InitDefaultExecEnv(void) {
  nnacl_default_env.Free = NNACLDefaultFree;
  nnacl_default_env.Alloc = NNACLDefaultAlloc;
  nnacl_default_env.ParallelLaunch = NNACLDefaultParallelLunch;
}

void CheckExecEnv(KernelBase *base) {
  if (base->env_ == NULL) {
    base->env_ = &nnacl_default_env;
  }
  return;
}
