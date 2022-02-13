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
#ifndef MINDSPORE_NNACL_INFER_INFER_REGISTER_H_
#define MINDSPORE_NNACL_INFER_INFER_REGISTER_H_

#include "nnacl/tensor_c.h"
#include "nnacl/op_base.h"
#include "nnacl/infer/infer.h"

#ifdef __cplusplus
extern "C" {
#endif

void RegInfer(int prim_type, InferShape func);

#ifdef _MSC_VER
#define REG_INFER(op, type, func)
#else
#define REG_INFER(op, type, func) \
  __attribute__((constructor(102))) void Reg##op##Infer() { RegInfer(type, func); }
#endif

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_NNACL_INFER_INFER_REGISTER_H_
