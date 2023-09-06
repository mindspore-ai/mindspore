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

#include "inc/sparse_segment_sqrt_n_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
graphStatus SparseSegmentReductionShapeFn(Operator &op) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeShape x_shapes;
  auto x_desc = op_desc->MutableInputDesc(0);
  if (WithRankAtLeast(x_desc, 1, x_shapes, op) != GRAPH_SUCCESS) {
    AICPU_OP_LOGE(TbeGetName(op).c_str(), "Input x should be at least 1-D.");
    return GRAPH_FAILED;
  }

  GeShape indices_shapes;
  auto indices_desc = op_desc->MutableInputDesc(1);
  if (WithRank(indices_desc, 1, indices_shapes, op) != GRAPH_SUCCESS) {
    AICPU_OP_LOGE(TbeGetName(op).c_str(), "Input indices must be 1-D.");
    return GRAPH_FAILED;
  }

  GeShape segment_ids_shapes;
  auto segment_ids_desc = op_desc->MutableInputDesc(2);
  if (WithRank(segment_ids_desc, 1, segment_ids_shapes, op) != GRAPH_SUCCESS) {
    AICPU_OP_LOGE(TbeGetName(op).c_str(), "Input segment_ids must be 1-D.");
    return GRAPH_FAILED;
  }

  GeShape unuse;
  if (Merge(indices_shapes, segment_ids_shapes, unuse, op) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  GeShape subshapes;
  if (SubShape(x_shapes, 1, x_shapes.GetDimNum(), 1, subshapes, op) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  std::vector<std::string> input_infer_depends = {"segment_ids"};
  op_desc->SetOpInferDepends(input_infer_depends);

  std::vector<int64_t> dims_0;
  Tensor segment_tensor;
  auto ret = op.GetInputConstData("segment_ids", segment_tensor);
  if (ret == GRAPH_SUCCESS) {
    Shape segment_data_shape = segment_tensor.GetTensorDesc().GetShape();
    int64_t segment_ShapeSize = segment_data_shape.GetShapeSize();
    if (segment_ids_desc->GetDataType() == DT_INT32) {
      auto max_segment_id = (reinterpret_cast<int32_t *>(segment_tensor.GetData()))[segment_ShapeSize - 1];
      dims_0.push_back(max_segment_id + 1);
    } else {
      auto max_segment_id = (reinterpret_cast<int64_t *>(segment_tensor.GetData()))[segment_ShapeSize - 1];
      dims_0.push_back(max_segment_id + 1);
    }
  } else {
    dims_0.push_back(ge::UNKNOWN_DIM);
  }
  GeShape dims_0_shape(dims_0);

  GeShape out;
  if (Concatenate(dims_0_shape, subshapes, out) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  auto y_desc = op_desc->MutableOutputDesc(0);
  y_desc->SetDataType(x_desc->GetDataType());
  y_desc->SetShape(out);

  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_INFERFUNC(SparseSegmentSqrtN, SparseSegmentSqrtNInfer) { return SparseSegmentReductionShapeFn(op); }

CUST_INFER_FUNC_REG(SparseSegmentSqrtN, SparseSegmentSqrtNInfer);
}  // namespace ge