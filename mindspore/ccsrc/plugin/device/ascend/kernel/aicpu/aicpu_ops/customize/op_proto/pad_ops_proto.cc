/**
 * Copyright (c) 2022-202 Huawei Technologies Co., Ltd.  All rights reserved.
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

#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

#include "op_proto/inc/pad_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"
#include "utils/error_util.h"
#include "utils/op_log.h"
#include "utils/op_const.h"

namespace ge {
namespace {
constexpr int kNum2 = 2;
}
// ----------------Pad Op Begin-------------------
static graphStatus PadInferShapeAndType(ge::Operator &op, std::vector<int64_t> &paddings) {
  static const int64_t input_x_idx = 0;
  auto input_desc = op.GetInputDesc(input_x_idx);
  const auto &input_shape = input_desc.GetShape();
  auto input_dtype = input_desc.GetDataType();
  static const int64_t output_y_idx = 0;
  auto output_desc = op.GetOutputDesc(output_y_idx);
  ge::Shape output_shape = output_desc.GetShape();
  output_desc.SetDataType(input_dtype);

  // input shape is -2, output is -2
  if (IsUnknownDimNum(input_shape)) {
    output_desc.SetShape(input_shape);
    op.UpdateOutputDesc("y", output_desc);
    return GRAPH_SUCCESS;
  }

  size_t dim_num = input_shape.GetDimNum();
  if (!IsUnknownShape(input_shape)) {
    // not dynamic shape, will output shape and dtype
    if (dim_num == 0) {
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("input shape cannot empty"));
      return GRAPH_FAILED;
    }
    if (dim_num * kNum2 != paddings.size()) {
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                          OtherErrMsg("the num of paddings must be double the input dim size"));
      return GRAPH_FAILED;
    }

    // calce the output shape
    output_shape = Shape(std::vector(dim_num, UNKNOWN_DIM));
    for (size_t dim = 0; dim < dim_num; dim++) {
      output_shape.SetDim(dim, input_shape.GetDim(dim) + paddings[dim * kNum2] + paddings[dim * kNum2 + 1]);
    }
    output_desc.SetShape(output_shape);
    op.UpdateOutputDesc("y", output_desc);
    return GRAPH_SUCCESS;
  }

  // input shape is -1, will get the shape and range
  // calcu the output shape
  output_shape = Shape(std::vector(dim_num, UNKNOWN_DIM));
  for (size_t dim = 0; dim < dim_num; dim++) {
    if (input_shape.GetDim(dim) == -1) {
      output_shape.SetDim(dim, input_shape.GetDim(dim));
    } else {
      output_shape.SetDim(dim, input_shape.GetDim(dim) + paddings[dim * kNum2] + paddings[dim * kNum2 + 1]);
    }
  }

  // calcu the output range
  std::vector<std::pair<int64_t, int64_t>> input_range;
  input_desc.GetShapeRange(input_range);
  MakeUpShapeRange(input_shape, input_range);
  std::vector<std::pair<int64_t, int64_t>> output_range;
  for (size_t dim = 0; dim < dim_num; dim++) {
    auto range_min = input_range[dim].first + paddings[dim * 2] + paddings[dim * 2 + 1];
    auto range_max =
      input_range[dim].second == -1 ? -1 : input_range[dim].second + paddings[dim * 2] + paddings[dim * 2 + 1];
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  }
  output_desc.SetShapeRange(output_range);
  output_desc.SetShape(output_shape);
  op.UpdateOutputDesc("y", output_desc);

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(PadInferShape) {
  OP_LOGD(TbeGetName(op), "InferShape Begin.");
  const vector<string> depend_names = {"paddings"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  // first get the padding const
  // get const paddings data
  std::vector<int64_t> paddings;
  if (!ops::GetConstIntData(op, "paddings", paddings)) {
    OP_LOGW(TbeGetName(op), "the node paddings is not const node, will set the output dynamic");
    auto input_desc = op.GetInputDesc("x");
    const auto &input_shape = input_desc.GetShape();
    DataType input_dtype = input_desc.GetDataType();
    auto output_desc = op.GetOutputDesc("y");

    // shape_x is UNKNOWN_RANK
    if (IsUnknownRankShape(input_shape)) {
      OP_LOGW(TbeGetName(op), "shape_x is UNKNOWN_RANK. Set output UNKNOWN_RANK");
      output_desc.SetShape(input_shape);
      output_desc.SetDataType(input_dtype);
      op.UpdateOutputDesc("y", output_desc);
      return GRAPH_SUCCESS;
    }

    size_t dim_num = input_shape.GetDimNum();
    // shape_x is UNKNOWN_DIM
    if (dim_num == 0) {
      dim_num = 1;
    }
    vector<int64_t> out_shape(dim_num, -1);
    std::vector<std::pair<int64_t, int64_t>> output_range;
    input_desc.GetShapeRange(output_range);
    MakeUpShapeRange(out_shape, output_range);
    for (size_t i = 0; i < dim_num; i++) {
      output_range[i].second = -1;
    }
    output_desc.SetShape(Shape(out_shape));
    output_desc.SetDataType(input_dtype);
    output_desc.SetShapeRange(output_range);
    op.UpdateOutputDesc("y", output_desc);
    return GRAPH_SUCCESS;
  }

  return PadInferShapeAndType(op, paddings);
}

COMMON_INFER_FUNC_REG(Pad, PadInferShape);
// ----------------Pad Op End-------------------
// ----------------PadV3 Op Begin-------------------
IMPLEMT_COMMON_INFERFUNC(PadV3InferShape) {
  const vector<string> depend_names = {"paddings"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  Tensor paddings_tensor;
  if (ge::GRAPH_SUCCESS != op.GetInputConstData("paddings", paddings_tensor)) {
    OP_LOGW(TbeGetName(op).c_str(), "Get Const Value [paddings] failed, Setting shape to UNKNOWN_DIM");
    Shape shape_x = op.GetInputDescByName("x").GetShape();
    vector<int64_t> shape;
    for (size_t dim = 0; dim < shape_x.GetDimNum(); dim++) {
      shape.push_back(UNKNOWN_DIM);
    }
    DataType input_dtype = op.GetInputDescByName("x").GetDataType();
    TensorDesc tensordesc_output = op.GetOutputDescByName("y");
    Shape out_shape(shape);
    tensordesc_output.SetShape(out_shape);
    tensordesc_output.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("y", tensordesc_output);
    return GRAPH_SUCCESS;
  }

  DataType dtype = op.GetInputDescByName("paddings").GetDataType();

  std::vector<int64_t> paddings;
  if (!GetConstValue(op, paddings_tensor, dtype, paddings)) {
    OP_LOGE(TbeGetName(op).c_str(), "Get Const Value [paddings] failed ");
    return GRAPH_FAILED;
  }

  bool paddings_contiguous = true;
  if (op.GetAttr("paddings_contiguous", paddings_contiguous) == GRAPH_FAILED) {
    OP_LOGI(TbeGetName(op).c_str(), "Get attr [paddings_contiguous] failed");
  }

  auto input_desc = op.GetInputDesc("x");
  auto input_shape = input_desc.GetShape().GetDims();
  auto input_shape_max = input_shape.size();
  auto paddings_shape = paddings.size();

  // expand paddings by 0
  auto expand_num = input_shape_max * 2 - paddings_shape;
  for (size_t dim = 0; dim < expand_num; dim++) {
    paddings.push_back(0);
  }

  if (expand_num > 0) {
    std::vector<int64_t> pad_vec;
    for (int i = static_cast<int>(input_shape_max); i > 0; i--) {
      pad_vec.push_back(paddings[i * 2 - 2]);
      pad_vec.push_back(paddings[i * 2 - 1]);
    }
    paddings = pad_vec;
  }

  if (!paddings_contiguous) {
    std::vector<int64_t> pads;
    int64_t rank = static_cast<int64_t>(paddings.size()) / 2;
    for (int i = 0; i < rank; i++) {
      pads.push_back(paddings[i]);
      pads.push_back(paddings[i + rank]);
    }
    paddings = pads;
    OP_LOGI(TbeGetName(op).c_str(), "Get attr paddings_contiguous = false");
  } else {
    OP_LOGI(TbeGetName(op).c_str(), "Get attr paddings_contiguous = true[default]");
  }

  return PadInferShapeAndType(op, paddings);
}

COMMON_INFER_FUNC_REG(PadV3, PadV3InferShape);
// ----------------PadV3 Op End-------------------

// ----------------PadV3Grad Op Begin-------------------
static graphStatus PadV3GradInferShapeAndType(ge::Operator &op, std::vector<int64_t> &paddings) {
  auto input_desc = op.GetInputDesc("x");
  auto input_shape = input_desc.GetShape().GetDims();
  auto input_dtype = input_desc.GetDataType();
  auto output_desc = op.GetOutputDesc("y");
  output_desc.SetDataType(input_dtype);

  if (!IsUnknown(input_shape)) {
    // calce the output shape
    vector<int64_t> output_shape;
    for (size_t dim = 0; dim < input_shape.size(); dim++) {
      output_shape.push_back(input_shape[dim] - paddings[dim * kNum2] - paddings[dim * kNum2 + 1]);
    }
    output_desc.SetShape(Shape(output_shape));
    op.UpdateOutputDesc("y", output_desc);
    return GRAPH_SUCCESS;
  }

  // input shape is -2, output is -2
  if (IsUnknownRankShape(input_shape)) {
    output_desc.SetShape(Shape(input_shape));
    op.UpdateOutputDesc("y", output_desc);

    return GRAPH_SUCCESS;
  }

  // input shape is -1, will get the shape and range
  // calcu the output shape
  vector<int64_t> output_shape;
  for (size_t dim = 0; dim < input_shape.size(); dim++) {
    if (input_shape[dim] == -1) {
      output_shape.push_back(input_shape[dim]);
    } else {
      output_shape.push_back(input_shape[dim] - paddings[dim * 2] - paddings[dim * 2 + 1]);
    }
  }
  output_desc.SetShape(Shape(output_shape));

  // calcu the output range
  std::vector<std::pair<int64_t, int64_t>> input_range;
  input_desc.GetShapeRange(input_range);
  MakeUpShapeRange(input_shape, input_range);
  std::vector<std::pair<int64_t, int64_t>> output_range;
  for (size_t dim = 0; dim < input_shape.size(); dim++) {
    auto range_min = input_range[dim].first - paddings[dim * 2] - paddings[dim * 2 + 1];
    if (range_min < 1) {
      range_min = 1;
    }
    auto range_max =
      input_range[dim].second == -1 ? -1 : input_range[dim].second - paddings[dim * 2] - paddings[dim * 2 + 1];
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  }
  output_desc.SetShapeRange(output_range);
  op.UpdateOutputDesc("y", output_desc);

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(PadV3GradInferShape) {
  const vector<string> depend_names = {"paddings"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  Tensor paddings_tensor;
  if (ge::GRAPH_SUCCESS != op.GetInputConstData("paddings", paddings_tensor)) {
    OP_LOGW(TbeGetName(op).c_str(), "Get Const Value [paddings] failed, Setting shape to UNKNOWN_DIM");
    Shape shape_x = op.GetInputDescByName("x").GetShape();
    vector<int64_t> shape;
    for (size_t dim = 0; dim < shape_x.GetDimNum(); dim++) {
      shape.push_back(UNKNOWN_DIM);
    }
    DataType input_dtype = op.GetInputDescByName("x").GetDataType();
    TensorDesc tensordesc_output = op.GetOutputDescByName("y");
    Shape out_shape(shape);
    tensordesc_output.SetShape(out_shape);
    tensordesc_output.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("y", tensordesc_output);
    return GRAPH_SUCCESS;
  }

  DataType dtype = op.GetInputDescByName("paddings").GetDataType();

  std::vector<int64_t> paddings;
  if (!GetConstValue(op, paddings_tensor, dtype, paddings)) {
    OP_LOGE(TbeGetName(op).c_str(), "Get Const Value [paddings] failed ");
    return GRAPH_FAILED;
  }

  bool paddings_contiguous = true;
  if (op.GetAttr("paddings_contiguous", paddings_contiguous) == GRAPH_FAILED) {
    OP_LOGI(TbeGetName(op).c_str(), "Get attr [paddings_contiguous] failed");
  }
  return PadV3GradInferShapeAndType(op, paddings);
}

COMMON_INFER_FUNC_REG(PadV3Grad, PadV3GradInferShape);
// ----------------PadV3Grad Op End-------------------
}  // namespace ge
