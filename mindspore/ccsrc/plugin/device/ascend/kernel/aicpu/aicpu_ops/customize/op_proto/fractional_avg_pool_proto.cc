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

#include "inc/ops/nn_pooling_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
IMPLEMT_INFERFUNC(FractionalAvgPool, FractionalAvgPoolInfer) {
  Shape input;
  if (WithRank(op.GetInputDesc(0), 4, input, op) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(
      TbeGetName(op), ConcatString("Call WithRank function failed, ",
                                   GetShapeErrMsg(0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "4D")));
    return GRAPH_FAILED;
  }
  std::vector<float> pooling_ratio;
  op.GetAttr("pooling_ratio", pooling_ratio);
  if (pooling_ratio.size() != 4) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       GetAttrValueErrMsg("pooling_ratio", ConcatString(pooling_ratio.size()), "4"));
    return GRAPH_PARAM_INVALID;
  }
  auto x_dims = op.GetInputDesc(0).GetShape().GetDims();
  std::vector<int64_t> dims;
  dims.reserve(4);
  for (int i = 0; i < 4; ++i) {
    int64_t val = ge::UNKNOWN_DIM;
    if (x_dims[i] != ge::UNKNOWN_DIM) {
      val = static_cast<int64_t>(x_dims[i] / pooling_ratio[i]);
      if (val < 0) {
        AICPU_INFER_SHAPE_INNER_ERR_REPORT(
          TbeGetName(op), ConcatString("size computed for ", i, "th dim is ", val, ", should be >= 0"));
        return GRAPH_PARAM_INVALID;
      }
    }

    OP_LOGI(TbeGetName(op).c_str(), "i = %d, x_dims[i] = %ld, pooling_ratio[i] = %f, val = %ld", i, x_dims[i],
            pooling_ratio[i], val);
    dims.push_back(val);
  }
  Shape out(dims);
  Shape row_pooling_sequence;
  (void)Vector(dims[1] + 1, row_pooling_sequence);
  Shape col_pooling_sequence;
  (void)Vector(dims[2] + 1, col_pooling_sequence);

  DataType type = op.GetInputDescByName("x").GetDataType();

  TensorDesc y_desc = op.GetOutputDescByName("y");
  y_desc.SetShape(out);
  y_desc.SetDataType(type);
  op.UpdateOutputDesc("y", y_desc);

  TensorDesc row_desc = op.GetOutputDescByName("row_pooling_sequence");
  row_desc.SetShape(row_pooling_sequence);
  row_desc.SetDataType(DT_INT64);
  op.UpdateOutputDesc("row_pooling_sequence", row_desc);

  TensorDesc col_desc = op.GetOutputDescByName("col_pooling_sequence");
  col_desc.SetShape(col_pooling_sequence);
  col_desc.SetDataType(DT_INT64);
  op.UpdateOutputDesc("col_pooling_sequence", col_desc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(FractionalAvgPool, FractionalAvgPoolInfer);
}  // namespace ge