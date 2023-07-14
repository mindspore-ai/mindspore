/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#include "runtime_util.h"
#include "utils/op_util.h"

using namespace ge;
namespace ops {
ge::graphStatus InferShape4Elewise(gert::InferShapeContext *context) {
  auto in_shape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape);
  auto out_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);

  if (IsUnknownRank(in_shape)) {
    OP_LOGD(context->GetNodeName(), "input shape is UnknownRank, set output shape to (-2, )");
    return SetUnknownRank(out_shape);
  }

  *out_shape = *in_shape;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CopyShapeInput2OutputWithIdx(gert::InferShapeContext *context, int64_t input_idx, int64_t output_idx) {
  auto in_shape = context->GetInputShape(input_idx);
  OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape);
  auto out_shape = context->GetOutputShape(output_idx);
  OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);
  *out_shape = *in_shape;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShape4InIdxAndOutVector(gert::InferShapeContext *context, int64_t input_idx,
                                             const std::vector<int64_t> &output_idxs) {
  auto in_shape = context->GetInputShape(input_idx);
  OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape);
  for (int64_t idx : output_idxs) {
    auto out_shape = context->GetOutputShape(idx);
    OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    *out_shape = *in_shape;
  }
  return ge::GRAPH_SUCCESS;
}

std::string ShapeCannotBroadcastMsg(const gert::Shape &shape1, const gert::Shape &shape2) {
  std::string res = "shape ";
  res += ToString(shape1);
  res += " and ";
  res += ToString(shape2);
  res += " cannot broadcast!";
  return res;
}

static bool BroadcastDim(int64_t &dim1, const int64_t dim2) {
  if (dim1 == dim2) {
    return true;
  }
  /* column is dim1, row is dim2, matrix value is broadcast(dim1, dim2)
  dim   0     1    d2
  0     0     0    E
  1     0     1    d2
  d1    E     d1   E
  */
  if ((dim1 != 1) && (dim2 != 1)) {
    string msg = ConcatString(dim1, " and ", dim2, " cannot broadcast!");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT("BroadcastDim", msg);
    return false;
  }
  dim1 = (dim1 == 1) ? dim2 : dim1;

  return true;
}

/*
 * @brief: broadcast new shape to output shape
 * @param [in] shape: const gert::Shape*, new shape to broadcast
 * @param [in/out] shape_output: gert::Shape*, output shape
 * @return succeed or not
 */
static bool BroadcastShapeToOutShape(const gert::Shape *shape, gert::Shape *shape_output) {
  OP_LOGD("BroadcastShapeToOutShape", "start broadcast %s to %s!", ToString(*shape).c_str(),
          ToString(*shape_output).c_str());
  size_t shape_len = shape->GetDimNum();
  size_t shape_y_len = shape_output->GetDimNum();
  if (shape_len > shape_y_len) {
    shape_output->SetDimNum(shape_len);
    size_t len_sub = shape_len - shape_y_len;
    for (size_t i = shape_y_len; i > 0; i--) {
      int64_t dim1 = shape->GetDim(len_sub + i - 1);
      int64_t dim2 = shape_output->GetDim(i - 1);
      if (!BroadcastDim(dim1, dim2)) {
        string msg = ConcatString(dim1, " and ", dim2, " cannot broadcast!");
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT("BroadcastShapeToOutShape", msg);
        return false;
      }
      shape_output->SetDim(len_sub + i - 1, dim1);
    }
    for (size_t i = 0; i < len_sub; i++) {
      shape_output->SetDim(i, shape->GetDim(i));
    }
  } else {
    auto len_sub = shape_y_len - shape_len;
    for (size_t i = 0; i < shape_len; i++) {
      int64_t dim1 = shape_output->GetDim(len_sub + i);
      int64_t dim2 = shape->GetDim(i);
      if (!BroadcastDim(dim1, dim2)) {
        string msg = ConcatString(dim1, " and ", dim2, " cannot broadcast!");
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT("BroadcastShapeToOutShape", msg);
        return false;
      }
      shape_output->SetDim(len_sub + i, dim1);
    }
  }
  return true;
}

bool BroadcastShape(const gert::Shape *in1_shape, const gert::Shape *in2_shape, gert::Shape *out_shape) {
  *out_shape = *in1_shape;

  OP_CHECK(!BroadcastShapeToOutShape(in2_shape, out_shape),
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT("BroadcastShape", ShapeCannotBroadcastMsg(*in2_shape, *in1_shape)),
           return false);
  return true;
}

bool BroadcastShape(const std::vector<const gert::Shape *> &in_shapes, gert::Shape *out_shape) {
  size_t size = in_shapes.size();
  OP_CHECK(size == 0, VECTOR_INFER_SHAPE_INNER_ERR_REPORT("BroadcastShape", "in_shapes is empty!"), return false);
  *out_shape = *in_shapes[0];

  for (size_t i = 1; i < size; i++) {
    OP_CHECK(!BroadcastShapeToOutShape(in_shapes[i], out_shape),
             VECTOR_INFER_SHAPE_INNER_ERR_REPORT("BroadcastShape", ShapeCannotBroadcastMsg(*in_shapes[i], *out_shape)),
             return false);
  }

  return true;
}

bool BroadcastShape(const gert::Shape **in_shapes, size_t size, gert::Shape *out_shape) {
  OP_CHECK(size == 0, VECTOR_INFER_SHAPE_INNER_ERR_REPORT("BroadcastShape", "in_shapes is empty!"), return false);
  *out_shape = *in_shapes[0];

  for (size_t i = 1; i < size; i++) {
    OP_CHECK(!BroadcastShapeToOutShape(in_shapes[i], out_shape),
             VECTOR_INFER_SHAPE_INNER_ERR_REPORT("BroadcastShape", ShapeCannotBroadcastMsg(*in_shapes[i], *out_shape)),
             return false);
  }

  return true;
}
}  // namespace ops
