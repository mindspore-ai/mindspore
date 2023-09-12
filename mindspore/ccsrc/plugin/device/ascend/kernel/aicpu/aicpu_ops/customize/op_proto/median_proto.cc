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

#include "inc/median_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
// ----------------Median-------------------
bool InferShapeAndTypeMedian(Operator &op, const string &input_name, const string &attr1_name, const string &attr2_name,
                             const string &attr3_name, const string &output_values, const string &output_indices) {
  Shape shape = op.GetInputDescByName(input_name.c_str()).GetShape();
  DataType input_dtype = op.GetInputDescByName(input_name.c_str()).GetDataType();
  Format input_format = op.GetInputDescByName(input_name.c_str()).GetFormat();
  std::vector<int64_t> dim_vec;
  std::vector<int64_t> dim_x = shape.GetDims();
  bool attr_global_median;
  int64_t attr_dim;
  bool attr_keepdim;
  if (ge::GRAPH_SUCCESS != op.GetAttr(attr1_name.c_str(), attr_global_median)) {
    std::string err_msg = GetInputInvalidErrMsg("global_median");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (attr_global_median == false) {
    if (ge::GRAPH_SUCCESS != op.GetAttr(attr2_name.c_str(), attr_dim)) attr_dim = 0;
    if (ge::GRAPH_SUCCESS != op.GetAttr(attr3_name.c_str(), attr_keepdim)) attr_keepdim = false;
    if (attr_dim < 0) {
      attr_dim += dim_x.size();
    }
    for (size_t i = 0; i < dim_x.size(); i++) {
      if (i != static_cast<size_t>(attr_dim)) {
        dim_vec.push_back(dim_x[i]);
      } else {
        if (attr_keepdim == true) {
          dim_vec.push_back(1);
        }
      }
    }
    ge::Shape output_shape = ge::Shape(dim_vec);
    TensorDesc values = op.GetOutputDescByName(output_values.c_str());
    values.SetShape(output_shape);
    values.SetDataType(input_dtype);
    values.SetFormat(input_format);
    (void)op.UpdateOutputDesc(output_values.c_str(), values);
    TensorDesc indices = op.GetOutputDescByName(output_indices.c_str());
    indices.SetShape(output_shape);
    indices.SetDataType({DT_INT64});
    indices.SetFormat(input_format);
    (void)op.UpdateOutputDesc(output_indices.c_str(), indices);
    return true;
  } else {
    std::vector<int64_t> dim_vec;
    Shape output_shape(dim_vec);
    TensorDesc values = op.GetOutputDescByName(output_values.c_str());
    values.SetShape(output_shape);
    values.SetDataType(input_dtype);
    values.SetFormat(input_format);
    (void)op.UpdateOutputDesc(output_values.c_str(), values);
    return true;
  }
}
IMPLEMT_COMMON_INFERFUNC(MedianInferShape) {
  if (InferShapeAndTypeMedian(op, "x", "global_median", "axis", "keepdim", "values", "indices")) {
    return GRAPH_SUCCESS;
  }
  OP_LOGE(TbeGetName(op), "Obtains the processing function of the output tensor fail!\n");
  return GRAPH_FAILED;
}

CUST_COMMON_INFER_FUNC_REG(Median, MedianInferShape);
// ----------------Median END-------------------
}  // namespace ge