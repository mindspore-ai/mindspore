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

#include "inc/sparse_tensor_to_csr_sparse_matrix_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
CUST_IMPLEMT_INFERFUNC(SparseTensorToCSRSparseMatrix, SparseTensorToCSRSparseMatrixInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  GeShape x_indices_shape;
  auto x_indices_desc = op_desc->MutableInputDesc(0);
  if (WithRank(x_indices_desc, 2, x_indices_shape, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Input x_indices_desc must be 2-D.");
    return GRAPH_FAILED;
  }

  GeShape x_values_shape;
  auto x_values_desc = op_desc->MutableInputDesc(1);
  if (WithRank(x_values_desc, 1, x_values_shape, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Input x_values must be 1-D.");
    return GRAPH_FAILED;
  }

  GeShape x_dense_shape_shape;
  auto x_dense_shape_desc = op_desc->MutableInputDesc(2);
  if (WithRank(x_dense_shape_desc, 1, x_dense_shape_shape, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Input x_dense_shape must be 1-D.");
    return GRAPH_FAILED;
  }

  GeShape unknown_dim_shape({ge::UNKNOWN_DIM});

  auto y_dense_shape_desc = op_desc->MutableOutputDesc(0);
  y_dense_shape_desc->SetDataType(x_dense_shape_desc->GetDataType());
  y_dense_shape_desc->SetShape(x_dense_shape_shape);

  auto y_batch_pointers_desc = op_desc->MutableOutputDesc(1);
  y_batch_pointers_desc->SetDataType(x_indices_desc->GetDataType());
  y_batch_pointers_desc->SetShape(unknown_dim_shape);

  auto y_row_pointers_desc = op_desc->MutableOutputDesc(2);
  y_row_pointers_desc->SetDataType(x_indices_desc->GetDataType());
  y_row_pointers_desc->SetShape(unknown_dim_shape);

  auto y_col_indices_desc = op_desc->MutableOutputDesc(3);
  y_col_indices_desc->SetDataType(x_indices_desc->GetDataType());
  y_col_indices_desc->SetShape(x_values_shape);

  auto y_values_desc = op_desc->MutableOutputDesc(4);
  y_values_desc->SetDataType(x_values_desc->GetDataType());
  y_values_desc->SetShape(x_values_shape);

  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(SparseTensorToCSRSparseMatrix, SparseTensorToCSRSparseMatrixInfer);
}  // namespace ge