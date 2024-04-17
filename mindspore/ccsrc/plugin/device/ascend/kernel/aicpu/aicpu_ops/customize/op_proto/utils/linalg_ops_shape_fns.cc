/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

/*!
 * \file linalg_ops_shape_fns.cpp
 * \brief
 */
#include "linalg_ops_shape_fns.h"
#include "op_log.h"
#include "common_shape_fns.h"

namespace ge {
constexpr int64_t kRnak = 2;
constexpr int64_t kEnd = -2;

graphStatus MakeBatchSquareMatrix(const TensorDesc &tensor, Shape &out, const ge::Operator &op) {
  Shape s;
  if (WithRankAtLeast(tensor, kRnak, s, op) == GRAPH_FAILED) {
    OP_LOGE(op, "input tensor's rank at least 2.");
    return GRAPH_FAILED;
  }
  size_t existing = s.GetDimNum();
  int64_t dim1 = s.GetDim(existing - 2);
  int64_t dim2 = s.GetDim(existing - 1);

  int64_t out_dim = 0;
  if (Merge(dim1, dim2, out_dim) == GRAPH_FAILED) {
    OP_LOGE(op, "Merge two dimension failed.");
    return GRAPH_FAILED;
  }

  Shape batch_shape;
  if (SubShape(s, 0, kEnd, 1, batch_shape, op) == GRAPH_FAILED) {
    OP_LOGE(op, "Get SubShape batch_shape Failed.");
    return GRAPH_FAILED;
  }
  if (Concatenate(batch_shape, Shape({out_dim, out_dim}), out) == GRAPH_FAILED) {
    OP_LOGE(op, "Concatenate batch_shape and out_dim Failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus MatrixSolve(const TensorDesc &tensor1, const TensorDesc &tensor2, bool square, Shape &out,
                        const ge::Operator &op) {
  Shape lhs;
  Shape rhs;
  if (square) {
    if (MakeBatchSquareMatrix(tensor1, lhs, op) == GRAPH_FAILED) {
      OP_LOGE(op, "MatrixSolve first input tensor Make Batch Square Matrix failed.");
      return GRAPH_FAILED;
    }
  } else {
    if (WithRankAtLeast(tensor1, kRnak, lhs, op) == GRAPH_FAILED) {
      OP_LOGE(op, "MatrixSolve func first input tensor must be at least 2.");
      return GRAPH_FAILED;
    }
  }
  if (WithRankAtLeast(tensor2, kRnak, rhs, op) == GRAPH_FAILED) {
    OP_LOGE(op, "MatrixSolve func second input tensor must be at least 2.");
    return GRAPH_FAILED;
  }

  Shape lhs_batch;
  Shape rhs_batch;
  // Make the common batch subshape.
  if (SubShape(lhs, 0, kEnd, 1, lhs_batch, op) == GRAPH_FAILED) {
    OP_LOGE(op, "SubShape lhs_batch in MatrixSolve func failed.");
    return GRAPH_FAILED;
  }
  if (SubShape(rhs, 0, kEnd, 1, rhs_batch, op) == GRAPH_FAILED) {
    OP_LOGE(op, "SubShape rhs_batch in MatrixSolve func failed.");
    return GRAPH_FAILED;
  }
  int64_t lhs_batch_dim;
  // Make sure the batch dimensions match between lhs and rhs.
  if (Merge(lhs_batch.GetDimNum(), rhs_batch.GetDimNum(), lhs_batch_dim) == GRAPH_FAILED) {
    OP_LOGE(op, "Merge dimension lhs_batch and rhs_batch failed.");
    return GRAPH_FAILED;
  }

  int64_t dim_val = 0;
  int64_t lhs_rank = static_cast<int64_t>(lhs.GetDimNum());
  int64_t rhs_rank = static_cast<int64_t>(rhs.GetDimNum());
  int64_t dim_lhs = lhs.GetDim(lhs_rank - 2);
  int64_t dim_rhs = rhs.GetDim(rhs_rank - 2);
  // lhs and rhs have the same value for m to be compatible.
  if (Merge(dim_lhs, dim_rhs, dim_val) == GRAPH_FAILED) {
    OP_LOGE(op, "Merge dimension dim_lhs and dim_rhs failed.");
    return GRAPH_FAILED;
  }
  int64_t dim_ret = lhs.GetDim(lhs_rank - 1);
  if (square) {
    if (Merge(dim_val, dim_ret, dim_ret) == GRAPH_FAILED) {
      OP_LOGE(op, "Merge dimension dim_val and dim_ret failed.");
      return GRAPH_FAILED;
    }
  }

  Shape s;
  // Build final shape (batch_shape + n + k) in <out>.
  if (Concatenate(lhs_batch, Shape({dim_ret}), s) == GRAPH_FAILED) {
    OP_LOGE(op, "Concatenate Two Shape failed.");
    return GRAPH_FAILED;
  }
  int64_t dims = rhs.GetDim(rhs_rank - 1);
  if (Concatenate(s, Shape({dims}), s) == GRAPH_FAILED) {
    OP_LOGE(op, "Concatenate Shape s and dims failed.");
    return GRAPH_FAILED;
  }
  out = s;
  return GRAPH_SUCCESS;
}
}  // namespace ge
