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

#include "inc/sparse_segment_sqrt_n_with_num_segments_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
CUST_IMPLEMT_INFERFUNC(SparseSegmentSqrtNWithNumSegments, SparseSegmentSqrtNWithNumSegmentsInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeShape x_shape;
  auto x_desc = op_desc->MutableInputDesc(0);
  if (WithRankAtLeast(x_desc, 1, x_shape, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Input x should be at least 1-D.");
    return GRAPH_FAILED;
  }

  GeShape indices_shape;
  auto indices_desc = op_desc->MutableInputDesc(1);
  if (WithRank(indices_desc, 1, indices_shape, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Input indices must be 1-D.");
    return GRAPH_FAILED;
  }

  GeShape segment_ids_shape;
  auto segment_ids_desc = op_desc->MutableInputDesc(2);
  if (WithRank(segment_ids_desc, 1, segment_ids_shape, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Input segment_ids must be 1-D.");
    return GRAPH_FAILED;
  }

  GeShape num_segments_shape;
  auto num_segments_desc = op_desc->MutableInputDesc(3);
  if (WithRankAtMost(num_segments_desc, 1, num_segments_shape, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Input nums_segments should be at most 1-D.");
    return GRAPH_FAILED;
  }

  GeShape unused;
  if (Merge(indices_shape, segment_ids_shape, unused, op) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  GeShape subshape;
  if (SubShape(x_shape, 1, x_shape.GetDimNum(), 1, subshape, op) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  Tensor tensor;
  int64_t nums = 0;
  if (op.GetInputConstData("num_segments", tensor) != GRAPH_SUCCESS) {
    nums = UNKNOWN_DIM;
  }

  if (nums != UNKNOWN_DIM) {
    if (MakeDimForScalarInput(tensor, nums, op) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("fail to get dim from tensor of input[num_segments]."));
      return GRAPH_FAILED;
    }
  }

  GeShape out;
  std::vector<int64_t> dims;
  dims.reserve(1);
  dims.push_back(nums);
  GeShape nums_shape(dims);
  if (Concatenate(nums_shape, subshape, out) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  auto y_desc = op_desc->MutableOutputDesc(0);
  y_desc->SetDataType(x_desc->GetDataType());
  y_desc->SetShape(out);

  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(SparseSegmentSqrtNWithNumSegments, SparseSegmentSqrtNWithNumSegmentsInfer);
}  // namespace ge