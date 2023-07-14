/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

#include "inc/csr_sparse_matrix_to_sparse_tensor_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
CUST_IMPLEMT_INFERFUNC(CSRSparseMatrixToSparseTensor, CSRSparseMatrixToSparseTensorInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  GeShape x_dense_shape_shape;
  auto x_dense_shape_desc = op_desc->MutableInputDesc(0);
  if (WithRank(x_dense_shape_desc, 1, x_dense_shape_shape, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Input x_dense_shape must be 1-D.");
    return GRAPH_FAILED;
  }

  GeShape x_batch_pointers_shape;
  auto x_batch_pointers_desc = op_desc->MutableInputDesc(1);
  if (WithRank(x_batch_pointers_desc, 1, x_batch_pointers_shape, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Input x_batch_pointers must be 1-D.");
    return GRAPH_FAILED;
  }

  GeShape x_row_pointers_shape;
  auto x_row_pointers_desc = op_desc->MutableInputDesc(2);
  if (WithRank(x_row_pointers_desc, 1, x_row_pointers_shape, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Input x_row_pointers must be 1-D.");
    return GRAPH_FAILED;
  }

  GeShape x_col_indices_shape;
  auto x_col_indices_desc = op_desc->MutableInputDesc(3);
  if (WithRank(x_col_indices_desc, 1, x_col_indices_shape, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Input x_col_indices must be 1-D.");
    return GRAPH_FAILED;
  }

  GeShape x_values_shape;
  auto x_values_desc = op_desc->MutableInputDesc(4);
  if (WithRank(x_values_desc, 1, x_values_shape, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Input x_values must be 1-D.");
    return GRAPH_FAILED;
  }

  GeShape unused;
  if (Merge(x_col_indices_shape, x_values_shape, unused, op) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  GeShape indices_shape;
  if (Concatenate(x_col_indices_shape, x_dense_shape_shape, indices_shape) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  auto indices_desc = op_desc->MutableOutputDesc(0);
  indices_desc->SetDataType(x_col_indices_desc->GetDataType());
  indices_desc->SetShape(indices_shape);

  auto values_desc = op_desc->MutableOutputDesc(1);
  values_desc->SetDataType(x_values_desc->GetDataType());
  values_desc->SetShape(x_values_shape);

  auto dense_shape_desc = op_desc->MutableOutputDesc(2);
  dense_shape_desc->SetDataType(x_dense_shape_desc->GetDataType());
  dense_shape_desc->SetShape(x_dense_shape_shape);

  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(CSRSparseMatrixToSparseTensor, CSRSparseMatrixToSparseTensorInfer);
}  // namespace ge