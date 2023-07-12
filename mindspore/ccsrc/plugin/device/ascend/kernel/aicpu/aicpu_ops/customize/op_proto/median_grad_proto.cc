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

#include "inc/median_grad_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
// ----------------MedianGrad Begin-------------------
IMPLEMT_COMMON_INFERFUNC(MedianGradInferShape) {
  TensorDesc out_desc = op.GetOutputDescByName("x_grad");
  Shape x_shape = op.GetInputDescByName("x").GetShape();
  Format x_format = op.GetInputDescByName("x").GetFormat();
  switch (op.GetInputDescByName("x").GetDataType()) {
    case (DT_INT16):
      out_desc.SetDataType(DT_FLOAT);
      break;
    case (DT_INT32):
      out_desc.SetDataType(DT_FLOAT);
      break;
    case (DT_INT64):
      out_desc.SetDataType(DT_FLOAT);
      break;
    case (DT_FLOAT):
      out_desc.SetDataType(DT_FLOAT);
      break;
    case (DT_DOUBLE):
      out_desc.SetDataType(DT_DOUBLE);
      break;
    default:
      return GRAPH_FAILED;
  }
  out_desc.SetShape(x_shape);
  out_desc.SetFormat(x_format);
  if (op.UpdateOutputDesc("x_grad", out_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(MedianGrad, MedianGradInferShape);
// ----------------MedianGrad End-------------------
}  // namespace ge