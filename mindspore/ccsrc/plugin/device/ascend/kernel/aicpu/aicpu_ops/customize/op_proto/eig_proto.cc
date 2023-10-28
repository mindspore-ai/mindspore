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

#include "inc/eig_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
// -----------------------Eig Starts----------------------
IMPLEMT_COMMON_INFERFUNC(EigInferShape) {
  Shape x_shape = op.GetInputDescByName("x").GetShape();
  DataType x_dtype = op.GetInputDescByName("x").GetDataType();
  DataType y_dtype;
  if (x_dtype != DT_FLOAT && x_dtype != DT_DOUBLE && x_dtype != DT_COMPLEX64 && x_dtype != DT_COMPLEX128) {
    OP_LOGE(TbeGetName(op).c_str(),
            "For Eig, input x's data type should be either of float, double, complex64, complex128.");
    return GRAPH_FAILED;
  }
  if (x_dtype == DT_FLOAT || x_dtype == DT_COMPLEX64) {
    y_dtype = DT_COMPLEX64;
  } else {
    y_dtype = DT_COMPLEX128;
  }
  bool compute_v;
  if (op.GetAttr("compute_v", compute_v) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "For Eig, get attr compute_v failed!");
    return GRAPH_FAILED;
  }
  int64_t rank = static_cast<int64_t>(x_shape.GetDimNum());
  vector<int64_t> x_shape_list = x_shape.GetDims();
  if (rank < 2) {
    OP_LOGE(TbeGetName(op).c_str(), "For Eig, rank of input x must be equal to or greater than 2.");
    return GRAPH_FAILED;
  }
  if (x_shape_list[rank - 1] != x_shape_list[rank - 2]) {
    OP_LOGE(TbeGetName(op).c_str(), "For Eig, input x must be square or batch squares.");
    return GRAPH_FAILED;
  }
  TensorDesc val_desc = op.GetOutputDescByName("eigen_values");
  TensorDesc vec_desc = op.GetOutputDescByName("eigen_vectors");
  val_desc.SetDataType(y_dtype);
  vec_desc.SetDataType(y_dtype);
  std::vector<int64_t> vec_shape_list = {};
  std::vector<int64_t> val_shape_list = {};
  val_shape_list.assign(x_shape_list.begin(), x_shape_list.end());
  val_shape_list.pop_back();
  if (compute_v) {
    vec_shape_list.assign(x_shape_list.begin(), x_shape_list.end());
  }
  Shape vec_shape(vec_shape_list);
  Shape val_shape(val_shape_list);
  vec_desc.SetShape(vec_shape);
  val_desc.SetShape(val_shape);
  if (op.UpdateOutputDesc("eigen_values", val_desc) != GRAPH_SUCCESS ||
      op.UpdateOutputDesc("eigen_vectors", vec_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Failed to update output desc.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(Eig, EigInferShape);
}  // namespace ge