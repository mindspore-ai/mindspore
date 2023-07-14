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

/*!
 * \file runtime_util.h
 * \brief
 */
#ifndef CUSTOMIZE_OP_PROTO_RUNTIME_RUNTIME_UTIL_H_
#define CUSTOMIZE_OP_PROTO_RUNTIME_RUNTIME_UTIL_H_

#include "utils/context_util.h"
#include "register/op_impl_registry.h"
#include "runtime/continuous_vector.h"
#include "runtime/infer_shape_context.h"
#include "runtime/storage_shape.h"
#include "error_util.h"
#include "op_util.h"

namespace ops {
using QuickVector = gert::Shape;
constexpr int64_t UNKNOWN_DIM_VALUE_ = -1;
constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2;

// Do infershape for OP which is single-input single-output and in-shape equal out-shape.
ge::graphStatus InferShape4Elewise(gert::InferShapeContext *context);

/*
 * @brief: get output shape
 * @param [in] context: gert::InferShapeContext
 * @param [in] input_idx: constvalue input index
 * @param [in] output_idx: constvalue output index
 * @return vector<int64_t>: success or failed
 */
ge::graphStatus CopyShapeInput2OutputWithIdx(gert::InferShapeContext *context, int64_t input_idx, int64_t output_idx);

/*
 * @brief: get output shape
 * @param [in] context: gert::InferShapeContext
 * @param [in] input_idx: constvalue input index
 * @param [in] output_idxs: constvalue output indexes,vector<int64_t>
 * @return graphStatus: success or failed
 */
ge::graphStatus InferShape4InIdxAndOutVector(gert::InferShapeContext *context, int64_t input_idx,
                                             const std::vector<int64_t> &output_idxs);

std::string ShapeCannotBroadcastMsg(const gert::Shape &shape1, const gert::Shape &shape2);
/*
 * @brief: broadcast new shape to output shape
 * @param [in] shape: const gert::Shape*, new shape to broadcast
 * @param [in/out] shape_output: gert::Shape*, output shape
 * @return succeed or not
 */
bool BroadcastShape(const gert::Shape *in1_shape, const gert::Shape *in2_shape, gert::Shape *out_shape);
bool BroadcastShape(const std::vector<const gert::Shape *> &in_shapes, gert::Shape *out_shape);
bool BroadcastShape(const gert::Shape **in_shapes, size_t size, gert::Shape *out_shape);

/*
 * @brief: set all the output shape to [-1, -1, ....] with input rank
 * @param [in] rank: the output input rank
 * @param [out] output_shape: the output shape ptr
 * @return ge::graphStatus
 */
inline ge::graphStatus SetAllUnknownDim(const int64_t rank, gert::Shape *output_shape) {
  OP_CHECK(output_shape == nullptr, OP_LOGD("SetAllUnknownDim", "the output_shape is nullptr, return failed"),
           return ge::GRAPH_FAILED);

  output_shape->SetDimNum(rank);
  for (int64_t i = 0; i < rank; ++i) {
    output_shape->SetDim(i, UNKNOWN_DIM_VALUE_);
  }
  OP_LOGD("SetAllUnknownDim", "set all dim = -1, output = %s", ToString(*output_shape).c_str());

  return ge::GRAPH_SUCCESS;
}

/*
 * @brief: set output shape to [-2]
 * @param [out] output_shape: the output shape ptr
 * @return ge::graphStatus
 */
inline ge::graphStatus SetUnknownRank(gert::Shape *output_shape) {
  OP_CHECK(output_shape == nullptr, OP_LOGD("SetUnknownRank", "the output_shape is nullptr, return failed"),
           return ge::GRAPH_FAILED);
  output_shape->SetDimNum(0);
  output_shape->AppendDim(UNKNOWN_RANK_DIM_VALUE_);

  OP_LOGD("SetUnknownRank", "set unknown rank = -2, output = %s", ToString(*output_shape).c_str());
  return ge::GRAPH_SUCCESS;
}

/*
 * @brief: check whether the output shape is unknown rank
 * @param [out] output_shape: the output shape ptr
 * @return ge::graphStatus
 */
inline bool IsUnknownRank(const gert::Shape *check_shape) {
  return check_shape->GetDimNum() == 1 && check_shape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE_;
}
}  // namespace ops

#endif  // CUSTOMIZE_OP_PROTO_RUNTIME_RUNTIME_UTIL_H_
