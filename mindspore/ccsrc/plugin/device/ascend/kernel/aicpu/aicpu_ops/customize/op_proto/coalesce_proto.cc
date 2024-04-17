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

#include "inc/coalesce_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
// ----------------Coalesce Begin------------------------
CUST_IMPLEMT_VERIFIER(Coalesce, CoalesceVerify) {
  TensorDesc x_indices_desc = op.GetInputDescByName("x_indices");
  TensorDesc x_values_desc = op.GetInputDescByName("x_values");
  TensorDesc x_shape_desc = op.GetInputDescByName("x_shape");
  DataType x_indices_type = x_indices_desc.GetDataType();
  DataType x_values_type = x_values_desc.GetDataType();
  DataType x_shape_type = x_shape_desc.GetDataType();
  Shape x_indices_shape = x_indices_desc.GetShape();
  Shape x_values_shape = x_values_desc.GetShape();
  Shape x_shape_shape = x_shape_desc.GetShape();
  if (x_indices_shape.GetDimNum() != 2) {
    OP_LOGE(TbeGetName(op).c_str(), "x_indices should be a 2-D tensor, but got x_indices is a %lu-D tensor.",
            x_indices_shape.GetDimNum());
    return GRAPH_FAILED;
  }
  if (x_values_shape.GetDimNum() != 1) {
    OP_LOGE(TbeGetName(op).c_str(), "x_values should be a 1-D tensor, but got x_values is a %lu-D tensor.",
            x_values_shape.GetDimNum());
    return GRAPH_FAILED;
  }
  if (x_shape_shape.GetDimNum() != 1) {
    OP_LOGE(TbeGetName(op).c_str(), "x_shape should be a 1-D tensor, but got x_shape is a %lu-D tensor.",
            x_shape_shape.GetDimNum());
    return GRAPH_FAILED;
  }
  if (x_indices_shape.GetDim(1) != x_values_shape.GetDim(0)) {
    OP_LOGE(TbeGetName(op).c_str(),
            "sizes of dim1 of x_indices and dim0 of x_values should be equal, but got %lu and %lu.",
            x_indices_shape.GetDim(1), x_values_shape.GetDim(0));
    return GRAPH_FAILED;
  }
  if (x_indices_shape.GetDim(0) != x_shape_shape.GetDim(0)) {
    OP_LOGE(TbeGetName(op).c_str(),
            "sizes of dim0 of x_indices and dim0 of x_shape should be equal, but got %lu and %lu.",
            x_indices_shape.GetDim(0), x_shape_shape.GetDim(0));
    return GRAPH_FAILED;
  }
  if (x_values_type != DT_DOUBLE && x_values_type != DT_FLOAT && x_values_type != DT_FLOAT16) {
    OP_LOGE(TbeGetName(op).c_str(), "the data type of x_values should be double, float or float16.");
    return GRAPH_FAILED;
  }
  if (x_indices_type != DT_INT64 || x_shape_type != DT_INT64) {
    OP_LOGE(TbeGetName(op).c_str(), "the data types of x_indices and x_shape should be int64.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(CoalesceInferShape) {
  TensorDesc x_indices_desc = op.GetInputDescByName("x_indices");
  TensorDesc x_values_desc = op.GetInputDescByName("x_values");
  TensorDesc x_shape_desc = op.GetInputDescByName("x_shape");
  DataType x_indices_type = x_indices_desc.GetDataType();
  DataType x_values_type = x_values_desc.GetDataType();
  DataType x_shape_type = x_shape_desc.GetDataType();
  Shape x_indices_shape = x_indices_desc.GetShape();
  Shape x_values_shape = x_values_desc.GetShape();
  Shape x_shape_shape = x_shape_desc.GetShape();
  TensorDesc y_indices_desc = op.GetOutputDescByName("y_indices");
  TensorDesc y_values_desc = op.GetOutputDescByName("y_values");
  TensorDesc y_shape_desc = op.GetOutputDescByName("y_shape");

  std::vector<int64_t> new_dims;
  new_dims.push_back(x_indices_shape.GetDim(0));
  new_dims.push_back(ge::UNKNOWN_DIM);
  y_indices_desc.SetShape(Shape(new_dims));
  y_indices_desc.SetDataType(x_indices_type);
  (void)op.UpdateOutputDesc("y_indices", y_indices_desc);
  new_dims.clear();
  new_dims.push_back(ge::UNKNOWN_DIM);
  y_values_desc.SetShape(Shape(new_dims));
  y_values_desc.SetDataType(x_values_type);
  (void)op.UpdateOutputDesc("y_values", y_values_desc);
  y_shape_desc.SetShape(x_shape_shape);
  y_shape_desc.SetDataType(x_shape_type);
  (void)op.UpdateOutputDesc("y_shape", y_shape_desc);
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(Coalesce, CoalesceInferShape);
CUST_VERIFY_FUNC_REG(Coalesce, CoalesceVerify);
// ----------------Coalesce END---------------------------------
}  // namespace ge
