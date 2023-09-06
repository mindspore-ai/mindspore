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
#include "utils/op_log.h"
#include "utils/op_const.h"

using namespace ge;
namespace ops {
// -------------------Diag Ops START---------------------
static constexpr size_t DIAG_IN_X_IDX = 0;
static constexpr size_t DIAG_OUT_Y_IDX = 0;
static constexpr size_t INT_DATA_2 = 2;

ge::graphStatus Infershape4Diag(gert::InferShapeContext *context) {
  OP_LOGD(context->GetNodeName(), "Begin to do DiagInfershape.");
  const gert::Shape *input_x_shape = context->GetInputShape(DIAG_IN_X_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, input_x_shape);
  gert::Shape *output_y_shape = context->GetOutputShape(DIAG_OUT_Y_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, output_y_shape);

  size_t x_dim_num = input_x_shape->GetDimNum();

  output_y_shape->SetDimNum(0);
  for (size_t i = 0; i < INT_DATA_2; i++) {
    for (size_t j = 0; j < x_dim_num; j++) {
      output_y_shape->AppendDim(input_x_shape->GetDim(j));
    }
  }

  OP_LOGD(context->GetNodeName(), "output_y_shape = %s.", ToString(*output_y_shape).c_str());
  OP_LOGD(context->GetNodeName(), "End to do DiagInfershape.");

  return ge::GRAPH_SUCCESS;
}

IMPL_OP(Diag).InferShape(Infershape4Diag);
// -------------------Diag Ops END---------------------

// -------------------DiagPart Ops START---------------------
ge::graphStatus Infershape4DiagPart(gert::InferShapeContext *context) {
  OP_LOGD(context->GetNodeName(), "Begin to do DiagPartInfershape.");
  const gert::Shape *input_x_shape = context->GetInputShape(DIAG_IN_X_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, input_x_shape);

  int64_t input_to_output_dims_times = 2;
  OP_CHECK(input_x_shape->GetDimNum() % 2 != 0,
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                                               "input_x_shape->GetDimNum() % 2 != 0 is not supported."),
           return ge::GRAPH_FAILED);
  int64_t output_shape_len = input_x_shape->GetDimNum() / input_to_output_dims_times;

  gert::Shape *output_y_shape = context->GetOutputShape(DIAG_OUT_Y_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, output_y_shape);

  output_y_shape->SetDimNum(output_shape_len);
  for (int64_t i = 0; i < output_shape_len; i++) {
    output_y_shape->SetDim(i, input_x_shape->GetDim(i));
  }

  OP_LOGD(context->GetNodeName(), "output_y_shape = %s.", ToString(*output_y_shape).c_str());
  OP_LOGD(context->GetNodeName(), "End to do DiagPartInfershape.");

  return ge::GRAPH_SUCCESS;
}

IMPL_OP(DiagPart).InferShape(Infershape4DiagPart);
// -------------------DiagPart Ops END---------------------
}  // namespace ops