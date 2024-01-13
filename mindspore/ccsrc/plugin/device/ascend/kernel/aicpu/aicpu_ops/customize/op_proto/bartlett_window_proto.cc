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

#include "custom_op_proto/cust_spectral_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
// ---------BartlettWindow-------------------
IMPLEMT_COMMON_INFERFUNC(BartlettWindowInferShape) {
  Operator::OpType dtype;
  Format input_format = op.GetInputDescByName("window_length").GetFormat();
  TensorDesc out_desc = op.GetOutputDescByName("y");
  Shape unused;
  std::string err_msg;

  // Set output shape
  if (WithRankAtMost(op.GetInputDescByName("window_length"), 1, unused, op) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(1, DebugString(op.GetInputDescByName("window_length").GetShape().GetDims()), "at most 1D");
    err_msg = std::string("failed to call WithRankAtMost, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<std::string> input_infer_depends = {"window_length"};
  PREPARE_DYNAMIC_SHAPE(input_infer_depends);
  Tensor tensor;
  Shape output_shape({UNKNOWN_DIM});
  if (op.GetInputConstData("window_length", tensor) == GRAPH_SUCCESS) {
    int64_t length;
    if (MakeDimForScalarInput(tensor, length, op) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }
    if (Vector(length, output_shape) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(),
                                         string("fail to gen vector shape according dim bins."));
      return GRAPH_FAILED;
    }
  }
  out_desc.SetShape(output_shape);

  // Set output dtype
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get attr dtype failed.");
    return GRAPH_FAILED;
  } else {
    switch (dtype) {
      case 0:
        out_desc.SetDataType(DT_FLOAT);
        break;
      case 1:
        out_desc.SetDataType(DT_FLOAT16);
        break;
      case 11:
        out_desc.SetDataType(DT_DOUBLE);
        break;
      default:
        return GRAPH_FAILED;
    }
  }

  // Set output format
  out_desc.SetFormat(input_format);

  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(BartlettWindow, BartlettWindowInferShape);
// ---------BartlettWindow End-------------------
}  // namespace ge