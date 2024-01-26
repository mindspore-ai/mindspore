/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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

#include "custom_op_proto/cust_math_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
IMPLEMT_COMMON_INFERFUNC(DCTInferShape) {
  Shape x_shape = op.GetInputDescByName("x").GetShape();
  DataType x_dtype = op.GetInputDescByName("x").GetDataType();
  DataType y_dtype;
  TensorDesc output_desc = op.GetOutputDescByName("y");

  // infer data type
  if (x_dtype == DT_DOUBLE || x_dtype == DT_COMPLEX128 || x_dtype == DT_INT64 || x_dtype == DT_UINT64) {
    y_dtype = DT_DOUBLE;
  } else {
    y_dtype = DT_FLOAT;
  }
  output_desc.SetDataType(y_dtype);

  int64_t type;
  if (op.GetAttr("type", type) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "For DCT, get attr type failed!");
    return GRAPH_FAILED;
  }
  int64_t n;
  if (op.GetAttr("n", n) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "For DCT, get attr n failed!");
    return GRAPH_FAILED;
  }
  int64_t axis;
  if (op.GetAttr("axis", axis) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "For DCT, get attr axis failed!");
    return GRAPH_FAILED;
  }
  int64_t rank = static_cast<int64_t>(x_shape.GetDimNum());
  vector<int64_t> x_shape_list = x_shape.GetDims();
  // check the value of axis and n
  if (axis < -rank || axis >= rank) {
    OP_LOGE(TbeGetName(op).c_str(), "For DCT, axis must be in range [-x.rank, x.rank).");
    return GRAPH_FAILED;
  }
  if (n != -1 && n < 0) {
    OP_LOGE(TbeGetName(op).c_str(), "For DCT, n must be greater than 0.");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> out_shape_list = {};
  out_shape_list.assign(x_shape_list.begin(), x_shape_list.end());
  if (n != -1) {
    auto positive_axis = axis < 0 ? axis + rank : axis;
    out_shape_list[positive_axis] = n;
  }

  Shape out_shape(out_shape_list);
  output_desc.SetShape(out_shape);
  if (op.UpdateOutputDesc("y", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Failed to update output desc.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(DCT, DCTInferShape);
}  // namespace ge