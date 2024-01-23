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

#include "custom_op_proto/cust_elewise_calculation_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/op_const.h"

namespace ge {
IMPLEMT_COMMON_INFERFUNC(ArgMaxInferShape) {
  // get all input desc
  const vector<string> depend_names = {"dimension"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  static const int64_t x_input_idx = 0;
  static const int64_t y_output_idx = 0;
  auto input_desc = op.GetInputDesc(x_input_idx);
  auto y_desc = op.GetOutputDesc(y_output_idx);
  // get x shape
  const auto &x_shape = input_desc.GetShape();

  // get and set output dtype
  ge::DataType dtype;
  if (op.GetAttr("dtype", dtype) == GRAPH_SUCCESS) {
    y_desc.SetDataType(dtype);
  } else {
    OP_LOGW(TbeGetName(op).c_str(), "get attr dtype failed.");
    y_desc.SetDataType(DT_INT32);
  }

  // if x_shape == -2, set output -2
  if (IsUnknownRankShape(x_shape)) {
    y_desc.SetShape(x_shape);
    return GRAPH_SUCCESS;
  }

  // if x_shape.size() < 2, set output scalar
  if (x_shape.GetDimNum() <= 1) {
    vector<int64_t> output_dims;
    y_desc.SetShape(Shape(output_dims));
    return GRAPH_SUCCESS;
  }

  // read dimension const value
  int64_t dimension = 0;
  if (ops::GetConstInt(op, "dimension", dimension)) {
    dimension = dimension < 0 ? dimension + static_cast<int64_t>(x_shape.GetDimNum()) : dimension;
    if ((dimension < 0) || (dimension >= static_cast<int64_t>(x_shape.GetDimNum()))) {
      OP_LOGE(TbeGetName(op), "The dimension value %ld must in range of input shape size %ld.", dimension,
              x_shape.GetDimNum());
      return GRAPH_FAILED;
    }

    ge::Shape output_shape = y_desc.GetShape();
    for (int64_t i = 0; i < dimension; i++) {
      output_shape.SetDim(i, x_shape.GetDim(i));
    }
    for (int64_t i = dimension + 1; i < static_cast<int64_t>(x_shape.GetDimNum()); i++) {
      output_shape.SetDim(i - 1, x_shape.GetDim(i));
    }

    // when output is dynamic will update range
    if (IsUnknownShape(output_shape)) {
      std::vector<std::pair<int64_t, int64_t>> input_range;
      input_desc.GetShapeRange(input_range);
      MakeUpShapeRange(x_shape, input_range);
      input_range.erase(input_range.begin() + dimension);
      y_desc.SetShapeRange(input_range);
    }
    y_desc.SetShape(output_shape);
    op.UpdateOutputDesc("y", y_desc);
    return GRAPH_SUCCESS;
  }

  // dimension is not const, set all output is -1, range is [0, -1]
  vector<int64_t> output_dims;
  std::vector<std::pair<int64_t, int64_t>> output_range;
  for (size_t item = 0; item < (x_shape.GetDimNum() - 1); ++item) {
    output_dims.push_back(-1);
  }
  MakeUpShapeRange(output_dims, output_range);
  y_desc.SetShape(Shape(output_dims));
  y_desc.SetShapeRange(output_range);
  op.UpdateOutputDesc("y", y_desc);

  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(ArgMax, ArgMaxInferShape);
}  // namespace ge