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
  TensorDesc v_output_desc = op.GetOutputDescByName("output");

  DataType input_dtype = op.GetInputDescByName("input").GetDataType();
  Format input_format = op.GetInputDescByName("input").GetFormat();
  ge::Shape shape_input = op.GetInputDescByName("input").GetShape();

  v_output_desc.SetShape(shape_input);
  v_output_desc.SetDataType(input_dtype);
  v_output_desc.SetFormat(input_format);

  if (op.UpdateOutputDesc("output", v_output_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(LogNormalReverse, LogNormalReverseInferShape);
// ----------------LogNormalReverse END-------------------

// ----------------Randperm-------------------
CUST_IMPLEMT_INFERFUNC(Randperm, RandpermInfer) {
  TensorDesc output_desc = op.GetOutputDescByName("output");
  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get attr dtype error.");
    return GRAPH_FAILED;
  }
  int64_t max_length;
  if (op.GetAttr("max_length", max_length) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get attr 'max_length' error.");
    return GRAPH_FAILED;
  }
  auto output_shape = Shape({max_length});
  output_desc.SetDataType(dtype);
  output_desc.SetShape(output_shape);
  return UpdateOutputDesc(op, output_desc);
}
CUST_INFER_FUNC_REG(Randperm, RandpermInfer);
// ----------------Randperm END-------------------

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

// ----------------Gamma-------------------
template <typename T>
static void GammaCaclDims(const Tensor &data, std::vector<int64_t> &vec_dim) {
  int32_t size = data.GetSize() / static_cast<int64_t>(sizeof(T));
  for (int32_t i = 0; i < size; i++) {
    T dim = *(reinterpret_cast<const T *>(data.GetData()) + i);
    vec_dim.push_back(dim);
  }
}

CUST_IMPLEMT_INFERFUNC(Gamma, GammaInfer) {
  const std::vector<std::string> depend_names = {"shape"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  Tensor shape_data;
  if (op.GetInputConstData("shape", shape_data) != GRAPH_SUCCESS) {
    OP_LOGI(TbeGetName(op).c_str(), "Get const value failed of [shape]");
    auto shape_desc = op.GetInputDesc("shape");
    std::vector<int64_t> shapedims = shape_desc.GetShape().GetDims();
    size_t dim_num = shapedims.size();

    if (dim_num > 1) {
      std::string err_msg = ConcatString("the rank[", dim_num, "] of input[shape] should not be more than 1");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }

    std::vector<int64_t> shape_vector(dim_num, -1);
    std::vector<std::pair<int64_t, int64_t>> range_vector(dim_num, std::make_pair(1, -1));

    auto output_desc = op.GetOutputDesc("output");
    output_desc.SetShape(Shape(shape_vector));
    output_desc.SetShapeRange(range_vector);
    output_desc.SetDataType(DT_FLOAT);
    op.UpdateOutputDesc("output", output_desc);
    return GRAPH_SUCCESS;
  }

  DataType shape_dtype = shape_data.GetTensorDesc().GetDataType();
  std::vector<int64_t> shape_dims;
  if (shape_dtype == DT_INT32) {
    GammaCaclDims<int32_t>(shape_data, shape_dims);
  } else if (shape_dtype == DT_INT64) {
    GammaCaclDims<int64_t>(shape_data, shape_dims);
  } else {
    std::string err_msg =
      ConcatString("dtype of input[shape] must be INT32 or INT64, but got [", DataTypeToStringDesc(shape_dtype), "].");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_PARAM_INVALID;
  }

  auto output_desc = op.GetOutputDesc("output");
  output_desc.SetShape(Shape(shape_dims));
  output_desc.SetDataType(DT_FLOAT);
  op.UpdateOutputDesc("output", output_desc);
  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(Gamma, GammaInfer);
// ----------------Gamma END-------------------
}  // namespace ge