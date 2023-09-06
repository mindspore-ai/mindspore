/**
 * Copyright (c) 2023 Huawei Technologies Co., Ltd.  All rights reserved.
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

#include "inc/ops/random_ops.h"
#include "inc/ops/stateful_random_ops.h"
#include "custom_op_proto/cust_random_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
IMPLEMT_INFERFUNC(NonDeterministicInts, NonDeterministicIntsInfer) {
  Shape shape;
  Tensor shape_tensor;
  if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get shape_tensor error.");
    return GRAPH_FAILED;
  }
  if (MakeShapeFromShapeTensor(shape_tensor, shape, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get shape error.");
    return GRAPH_FAILED;
  }

  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get attr dtype error.");
    return GRAPH_FAILED;
  }
  TensorDesc outputDesc = op.GetOutputDescByName("y");
  outputDesc.SetDataType(dtype);
  outputDesc.SetShape(shape);
  return op.UpdateOutputDesc("y", outputDesc);
}

INFER_FUNC_REG(NonDeterministicInts, NonDeterministicIntsInfer);

// ----------------LogNormalReverse-------------------
// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(LogNormalReverseInferShape) {
  TensorDesc v_output_desc = op.GetOutputDescByName("y");

  DataType input_dtype = op.GetInputDescByName("x").GetDataType();
  Format input_format = op.GetInputDescByName("x").GetFormat();
  ge::Shape shape_input = op.GetInputDescByName("x").GetShape();

  v_output_desc.SetShape(shape_input);
  v_output_desc.SetDataType(input_dtype);
  v_output_desc.SetFormat(input_format);

  if (op.UpdateOutputDesc("y", v_output_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(LogNormalReverse, LogNormalReverseInferShape);
// ----------------LogNormalReverse END-------------------

// ----------------Dropout2D-------------------
IMPLEMT_COMMON_INFERFUNC(Dropout2DInferShape) {
  TensorDesc output_desc = op.GetOutputDescByName("output");
  TensorDesc mask_desc = op.GetOutputDescByName("mask");

  DataType input_dtype = op.GetInputDescByName("x").GetDataType();
  ge::Shape shape_input = op.GetInputDescByName("x").GetShape();

  output_desc.SetShape(shape_input);
  output_desc.SetDataType(input_dtype);
  mask_desc.SetShape(shape_input);
  mask_desc.SetDataType(DT_BOOL);

  if (op.UpdateOutputDesc("output", output_desc) != GRAPH_SUCCESS ||
      op.UpdateOutputDesc("mask", mask_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(Dropout2D, Dropout2DInferShape);
// ----------------Dropout2D END-------------------
}  // namespace ge