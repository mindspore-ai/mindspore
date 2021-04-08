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
#ifndef MINDSPORE_NNACL_FP32_SPARSETODENSE_H_
#define MINDSPORE_NNACL_FP32_SPARSETODENSE_H_

#include "nnacl/sparse_to_dense_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
void SparseToDense(int **sparse_indices_vect, const int *output_shape, const float *sparse_values, float default_value,
                   float *output, bool isScalar, int index_start, int index_end, int out_width);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_SPARSETODENSE_H_
