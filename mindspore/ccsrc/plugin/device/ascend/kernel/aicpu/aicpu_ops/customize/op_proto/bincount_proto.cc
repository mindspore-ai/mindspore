/**
 * Copyright (c) 2023 Huawei Technologies Co., Ltd.  All rights reserved.
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

#include "op_proto/inc/math_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
IMPLEMT_INFERFUNC(Bincount, BincountInfer) {
  SetOpInferDepends(op, {"size"});

  Shape unused;
  auto size_desc = op.GetInputDesc(1);
  if (WithRank(size_desc, 0, unused, op) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1, DebugString(size_desc.GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  Tensor tensor;
  int64_t bins = 0;
  if (op.GetInputConstData("size", tensor) != GRAPH_SUCCESS) {
    bins = UNKNOWN_DIM;
  }

  if (bins != UNKNOWN_DIM) {
    if (MakeDimForScalarInput(tensor, bins, op) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("fail to get dim from tensor of input[size]."));
      return GRAPH_FAILED;
    }
  }

  Shape bins_shape;
  if (Vector(bins, bins_shape) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("fail to gen vector shape according dim bins."));
    return GRAPH_FAILED;
  }

  auto bins_desc = op.GetOutputDesc(0);
  bins_desc.SetShape(Shape(bins_shape.GetDims()));
  bins_desc.SetDataType(op.GetInputDesc(2).GetDataType());
  op.UpdateOutputDesc("bins", bins_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Bincount, BincountInfer);
}  // namespace ge