/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "nnacl/infer/infer_register.h"
#ifdef MS_COMPILE_IOS
extern void _ReducePrimType_ReduceFusion();
extern void _ReshapePrimType_Reshape();

void RegisterInfer() {
  _ReducePrimType_ReduceFusion();
  _ReshapePrimType_Reshape();
}
#endif
InferShape *g_infer_func = NULL;

__attribute__((constructor(101))) void InitInferFuncBuf() {
  if (g_infer_func != NULL) {
    return;
  }
  g_infer_func = malloc(PrimType_MAX * sizeof(InferShape));
  if (g_infer_func != NULL) {
    memset(g_infer_func, 0, PrimType_MAX * sizeof(InferShape));
  }
#ifdef MS_COMPILE_IOS
  RegisterInfer();
#endif
}

__attribute__((destructor)) void DestroyInferFuncBuf() {
  if (g_infer_func == NULL) {
    return;
  }
  free(g_infer_func);
  g_infer_func = NULL;
}

InferShape GetInferFunc(int prim_type) {
  if (g_infer_func != NULL && prim_type < PrimType_MAX) {
    return g_infer_func[prim_type];
  }
  return NULL;
}

void RegInfer(int prim_type, InferShape func) {
  if (g_infer_func != NULL && prim_type < PrimType_MAX) {
    g_infer_func[prim_type] = func;
  }
}
