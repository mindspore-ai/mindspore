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

#include "inc/ops/nn_pooling_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
IMPLEMT_INFERFUNC(FractionalAvgPoolGrad, FractionalAvgPoolGradInfer) {
  Tensor tensor;
  if (op.GetInputConstData("orig_input_tensor_shape", tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       ConcatString("get const data from input[orig_input_tensor_shape] failed"));
    return GRAPH_FAILED;
  }

  Shape result;
  if (MakeShapeFromShapeTensor(tensor, result, op) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
                                      ConcatString("call MakeShapeFromShapeTensor function failed to make",
                                                   " shape from input[orig_input_tensor_shape] data"));
    return GRAPH_FAILED;
  }

  DataType y_type = op.GetInputDescByName("out_backprop").GetDataType();
  TensorDesc out_desc = op.GetOutputDescByName("y");
  out_desc.SetShape(Shape(result));
  out_desc.SetDataType(y_type);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("update output[y] desc failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(FractionalAvgPoolGrad, FractionalAvgPoolGradInfer);
}  // namespace ge