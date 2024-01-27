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

#include "op_proto/inc/elewise_calculation_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/op_const.h"

namespace ge {
IMPLEMT_COMMON_INFERFUNC(ArgMinInferShape) {
  // get all input desc
  const vector<string> depend_names = {"dimension"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  auto input_desc = op.GetInputDesc("x");
  auto const_desc = op.GetInputDesc("dimension");
  auto y_desc = op.GetOutputDesc("y");

  // get and set output dtype
  ge::DataType dtype;
  if (op.GetAttr("dtype", dtype) == GRAPH_SUCCESS) {
    y_desc.SetDataType(dtype);
  } else {
    OP_LOGW(TbeGetName(op).c_str(), "get attr dtype failed.");
    y_desc.SetDataType(DT_INT32);
  }

  // get x shape
  auto x_shape = input_desc.GetShape().GetDims();
  // if x_shape == -2, set output -2
  if (IsUnknownRankShape(x_shape)) {
    y_desc.SetShape(Shape(x_shape));
    op.UpdateOutputDesc("y", y_desc);
    return GRAPH_SUCCESS;
  }

  // if x_shape.size() < 2, set output scalar
  if (x_shape.size() < 2) {
    vector<int64_t> output_shape;
    y_desc.SetShape(Shape(output_shape));
    op.UpdateOutputDesc("y", y_desc);
    return GRAPH_SUCCESS;
  }

  // read dimension const value
  vector<int64_t> dimension_value;
  Tensor dimension_tensor;
  if (op.GetInputConstData("dimension", dimension_tensor) == GRAPH_SUCCESS) {
    auto const_dtype = const_desc.GetDataType();
    GetConstValue(op, dimension_tensor, const_dtype, dimension_value);
    // verify dimension_value
    if (dimension_value.size() != 1) {
      string error_msg = ConcatString("the element size of input[dimension] should be equal to 1, but get ",
                                      dimension_value.size(), ".");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
      return GRAPH_FAILED;
    }
    int64_t dimension =
      dimension_value[0] < 0 ? dimension_value[0] + static_cast<int64_t>(x_shape.size()) : dimension_value[0];
    if (dimension >= static_cast<int64_t>(x_shape.size())) {
      string error_msg = ConcatString("the value of input[dimension] must be range at input shape size,",
                                      " but get input[dimension] value ", dimension_value[0], ", input[x] shape size ",
                                      x_shape.size(), ".");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
      return GRAPH_FAILED;
    }

    vector<int64_t> output_shape(x_shape);
    output_shape.erase(output_shape.begin() + dimension);
    y_desc.SetShape(Shape(output_shape));

    // when output is dynamic will update range
    if (IsUnknown(output_shape)) {
      std::vector<std::pair<int64_t, int64_t>> input_range;
      input_desc.GetShapeRange(input_range);
      MakeUpShapeRange(x_shape, input_range);
      input_range.erase(input_range.begin() + dimension);
      y_desc.SetShapeRange(input_range);
    }
    return GRAPH_SUCCESS;
  }

  // dimension is not const, set all output is -1, range is [1, -1]
  std::vector<std::pair<int64_t, int64_t>> output_range;
  vector<int64_t> output_shape;
  for (size_t item = 0; item < (x_shape.size() - 1); ++item) {
    output_shape.push_back(-1);
  }
  MakeUpShapeRange(output_shape, output_range);
  y_desc.SetShape(Shape(output_shape));
  y_desc.SetShapeRange(output_range);
  op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ArgMin, ArgMinInferShape);
}  // namespace ge