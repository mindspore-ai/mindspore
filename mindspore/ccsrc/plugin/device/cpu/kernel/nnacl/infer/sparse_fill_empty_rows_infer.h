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
#ifndef MINDSPORE_NNACL_SPARSE_FILL_EMPTY_ROWS_H
#define MINDSPORE_NNACL_SPARSE_FILL_EMPTY_ROWS_H

#include "nnacl/infer/common_infer.h"

#ifdef __cplusplus
extern "C" {
#endif

int SparseFillEmptyRowsInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs,
                                  size_t outputs_size, OpParameter *parameter);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_NNACL_SPARSE_FILL_EMPTY_ROWS_H
