/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/sparse_ops_declare.h"
#include "mindspore/core/ops/sparse_ops.h"

namespace mindspore::transform {
// CSRSparseMatrixToSparseTensor
CUST_INPUT_MAP(CSRSparseMatrixToSparseTensor) = {{1, INPUT_DESC(x_dense_shape)},
                                                 {2, INPUT_DESC(x_batch_pointers)},
                                                 {3, INPUT_DESC(x_row_pointers)},
                                                 {4, INPUT_DESC(x_col_indices)},
                                                 {5, INPUT_DESC(x_values)}};
CUST_ATTR_MAP(CSRSparseMatrixToSparseTensor) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(CSRSparseMatrixToSparseTensor) = {
  {0, OUTPUT_DESC(indices)}, {1, OUTPUT_DESC(values)}, {2, OUTPUT_DESC(dense_shape)}};
REG_ADPT_DESC(CSRSparseMatrixToSparseTensor, prim::kPrimCSRSparseMatrixToSparseTensor->name(),
              CUST_ADPT_DESC(CSRSparseMatrixToSparseTensor));
}  // namespace mindspore::transform
