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

InferShape g_infer_func[PrimType_MAX];

InferShape GetInferFunc(int prim_type) {
  if (prim_type < PrimType_MAX) {
    return g_infer_func[prim_type];
  }
  return NULL;
}

void RegInfer(int prim_type, InferShape func) {
  if (prim_type < PrimType_MAX) {
    g_infer_func[prim_type] = func;
  }
}
