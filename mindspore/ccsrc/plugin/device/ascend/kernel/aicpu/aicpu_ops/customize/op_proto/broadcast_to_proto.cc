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

#include "inc/ops/pad_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"
namespace ge {
template <typename T>
static void CaclDims(const Tensor &data, std::vector<int64_t> &vec_dim) {
  int32_t size = data.GetSize() / sizeof(T);
  for (int32_t i = 0; i < size; i++) {
    T dim = *((T *)data.GetData() + i);
    vec_dim.push_back(dim);
  }
}

// -------------------BroadcastTo-----------------------
IMPLEMT_INFERFUNC(BroadcastTo, BroadcastToInferShape) {
  const vector<string> depend_names = {"shape"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  Tensor data;
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  if (op.GetInputConstData("shape", data) != GRAPH_SUCCESS) {
    OP_LOGI(TbeGetName(op).c_str(), "Get constValue failed of [shape]");
    auto shape_desc = op_info->MutableInputDesc("shape");
    vector<int64_t> shapedims = shape_desc->MutableShape().GetDims();
    size_t dim_num = shapedims.size();

    DataType input_dtype = op.GetInputDescByName("x").GetDataType();

    if (dim_num > 1) {
      std::string err_msg = ConcatString("the rank[", dim_num, "] of input[shape] should not be more than 1");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }

    std::vector<int64_t> shape_vector;
    std::vector<std::pair<int64_t, int64_t>> range_vector;
    for (int64_t item = 0; item < shapedims[0]; ++item) {
      shape_vector.push_back(-1);
      range_vector.push_back(std::make_pair(1, -1));
    }
    auto output_desc = op_info->MutableOutputDesc("y");
    output_desc->SetShape(GeShape(shape_vector));
    output_desc->SetShapeRange(range_vector);
    output_desc->SetDataType(input_dtype);
    return GRAPH_SUCCESS;
  }

  DataType data_type = data.GetTensorDesc().GetDataType();
  std::vector<int64_t> vec_dim;
  if (data_type == DT_INT32) {
    CaclDims<int32_t>(data, vec_dim);
  } else if (data_type == DT_INT64) {
    CaclDims<int64_t>(data, vec_dim);
  } else {
    return GRAPH_PARAM_INVALID;
  }
  OP_LOGI(TbeGetName(op).c_str(), "the op infer shape and dtype");
  DataType input_dtype = op.GetInputDescByName("x").GetDataType();

  auto output_desc = op_info->MutableOutputDesc("y");
  output_desc->SetShape(GeShape(vec_dim));
  output_desc->SetDataType(input_dtype);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BroadcastTo, BroadcastToInferShape);
// --------------------BroadcastTo END-----------------------
}  // namespace ge