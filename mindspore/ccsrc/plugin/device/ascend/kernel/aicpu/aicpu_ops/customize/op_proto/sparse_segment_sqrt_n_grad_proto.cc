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

#include "inc/sparse_segment_sqrt_n_grad_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
CUST_IMPLEMT_INFERFUNC(SparseSegmentSqrtNGrad, SparseSegmentSqrtNGradInfer) {
  std::string err_msg;

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  std::vector<std::string> input_infer_depends = {"output_dim0"};
  op_desc->SetOpInferDepends(input_infer_depends);

  auto x_desc = op_desc->MutableInputDesc(0);
  GeShape x_ge_shape;
  if (WithRankAtLeast(x_desc, 1, x_ge_shape, op) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(0, DebugString(x_desc->GetShape().GetDims()), "at least 1D");
    err_msg = string("failed to call WithRankAtLeast function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto indices_desc = op_desc->MutableInputDesc(1);
  GeShape indices_shape;
  if (WithRank(indices_desc, 1, indices_shape, op) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(1, DebugString(indices_desc->GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  GeShape unused;
  GeShape segment_ids_shape(op_desc->MutableInputDesc(2)->GetShape());
  if (Merge(segment_ids_shape, indices_shape, unused, op) != GRAPH_SUCCESS) {
    err_msg = ConcatString("failed to call Merge function to merge input[segment_ids]'s shape",
                           DebugString(op_desc->MutableInputDesc(2)->GetShape().GetDims()),
                           " and input[indices]'s shape", DebugString(indices_shape.GetDims()));
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  auto unused_desc = op_desc->MutableInputDesc(3);
  if (WithRank(unused_desc, 0, unused, op) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(3, DebugString(unused_desc->GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto x_shape_dims = x_ge_shape.GetDims();
  Shape x_shape(x_shape_dims);
  Shape subshape;
  if (SubShape(x_shape, 1, x_shape.GetDimNum(), 1, subshape, op) != GRAPH_SUCCESS) {
    err_msg = ConcatString("failed to call SubShape function to get subshape from ", x_shape.GetDimNum(),
                           " to 1 in input[x] shape", DebugString(x_shape.GetDims()));
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  Tensor dims0_tensor;
  Shape dim0_shape;
  const int32_t *dims0_data;
  if (op.GetInputConstData("output_dim0", dims0_tensor) == GRAPH_SUCCESS) {
    void *dims0 = dims0_tensor.GetData();
    dims0_data = static_cast<const int32_t *>(dims0);
  } else {
    dims0_data = reinterpret_cast<const int32_t *>(&UNKNOWN_DIM);
  }

  dim0_shape = Shape({*dims0_data});

  Shape out;
  if (Concatenate(dim0_shape, subshape, out) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  auto out_dims = out.GetDims();
  GeShape ge_out(out_dims);
  auto out_desc = op_desc->MutableOutputDesc(0);
  out_desc->SetDataType(x_desc->GetDataType());
  out_desc->SetShape(ge_out);

  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(SparseSegmentSqrtNGrad, SparseSegmentSqrtNGradInfer);
}  // namespace ge