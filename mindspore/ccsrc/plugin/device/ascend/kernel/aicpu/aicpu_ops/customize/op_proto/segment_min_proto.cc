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

#include "inc/segment_min_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"
namespace ge {
static bool SegmentShapeVerify(const Operator &op, const std::string &input_name, const std::string &segment_ids_name) {
  auto input_shape_dims = op.GetInputDescByName("x").GetShape().GetDims();
  auto segment_ids_shape_dims = op.GetInputDescByName("segment_ids").GetShape().GetDims();
  if (input_shape_dims.empty() || segment_ids_shape_dims.empty()) {
    OP_LOGE(TbeGetName(op).c_str(), "shape of input is empty.");
    return false;
  }

  return true;
}
// --------------SegmentMin-----------------------
CUST_IMPLEMT_VERIFIER(SegmentMin, SegmentMinVerify) {
  if (!SegmentShapeVerify(op, "x", "segment_ids")) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SegmentMinInferShape) {
  auto input_desc = op.GetInputDescByName("x");
  const std::string segment_ids_name = "segment_ids";
  Tensor segment_ids;
  int64_t first_axis_dims;
  if (GRAPH_SUCCESS != op.GetInputConstData(segment_ids_name.c_str(), segment_ids)) {
    OP_LOGI("segment_min", "GetInputConstData %s failed.", segment_ids_name.c_str());
    first_axis_dims = -1;
  } else {
    auto data_type = op.GetInputDescByName(segment_ids_name.c_str()).GetDataType();
    std::vector<int64_t> const_data;
    if (!GetConstIntData(segment_ids, data_type, const_data)) {
      std::string err_msg =
        ConcatString("failed to call GetConstIntData function ",
                     "due to invalid data type of input[segment_ids]. data_type is ", DTypeStr(data_type));
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
    first_axis_dims = (*std::max_element(const_data.begin(), const_data.end())) + 1;
  }

  auto output_shape_dims = input_desc.GetShape().GetDims();
  if (output_shape_dims.empty()) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("the input[x]'s shape should not be empty."));
    return GRAPH_FAILED;
  }
  output_shape_dims[0] = first_axis_dims;
  Shape output_shape(output_shape_dims);
  DataType input_dtype = input_desc.GetDataType();
  TensorDesc tensor_desc_output = op.GetOutputDescByName("y");
  tensor_desc_output.SetShape(output_shape);
  tensor_desc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", tensor_desc_output);
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(SegmentMin, SegmentMinInferShape);
CUST_VERIFY_FUNC_REG(SegmentMin, SegmentMinVerify);
// ----------------SegmentMin END-------------------
}  // namespace ge