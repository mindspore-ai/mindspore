/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_NNACL_TENSORC_UTILS_H_
#define MINDSPORE_NNACL_TENSORC_UTILS_H_

#include <stddef.h>
#include "nnacl/errorcode.h"
#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"

#ifdef __cplusplus
extern "C" {
#endif

int GetBatch(const TensorC *tensor);
int GetHeight(const TensorC *tensor);
int GetWidth(const TensorC *tensor);
int GetChannel(const TensorC *tensor);
void SetBatch(TensorC *tensor, int batch);
void SetHeight(TensorC *tensor, int height);
void SetWidth(TensorC *tensor, int width);
void SetChannel(TensorC *tensor, int channel);
int GetElementNum(const TensorC *tensor);
int GetSize(const TensorC *tensor);
int GetDimensionSize(const TensorC *tensor, const size_t index);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_TENSORC_UTILS_H_
