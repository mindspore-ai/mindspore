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

#include "inc/blackman_window_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
// ----------------BlackmanWindow------------------
CUST_IMPLEMT_VERIFIER(BlackmanWindow, BlackmanWindowVerify) { return GRAPH_SUCCESS; }

IMPLEMT_COMMON_INFERFUNC(BlackmanWindowInferShape) {
  Shape shape;
  Shape unused;

  if (WithRank(op.GetInputDesc(0), 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  Tensor window_length_tensor;
  if (op.GetInputConstData("window_length", window_length_tensor) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  int64_t length;
  if (MakeDimForScalarInput(window_length_tensor, length, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  if (Vector(length, shape) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), string("fail to gen vector shape according dim bins."));
    return GRAPH_FAILED;
  }
  Operator::OpType type;
  if (op.GetAttr("dtype", type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), std::string("get attr[dtype] failed"));
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDescByName("y");
  output_desc.SetDataType(type);
  output_desc.SetShape(shape);
  return op.UpdateOutputDesc("y", output_desc);
}

CUST_COMMON_INFER_FUNC_REG(BlackmanWindow, BlackmanWindowInferShape);
CUST_VERIFY_FUNC_REG(BlackmanWindow, BlackmanWindowVerify);
// ----------------BlackmanWindow End----------------------
}  // namespace ge