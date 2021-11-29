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
#ifndef MINDSPORE_NNACL_FP16_SPARSE_TO_DENSE_FP16_H_
#define MINDSPORE_NNACL_FP16_SPARSE_TO_DENSE_FP16_H_

#include "nnacl/sparse_to_dense_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
int SparseToDenseSetDefaultFp16(float16_t *output, float16_t default_value, SparseToDenseParameter *param, int task_id);
int SparseToDenseFp16(int *indices_vec, const float16_t *sparse_values, float16_t default_value, float16_t *output,
                      SparseToDenseParameter *param, int task_id);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP16_SPARSE_TO_DENSE_FP16_H_
