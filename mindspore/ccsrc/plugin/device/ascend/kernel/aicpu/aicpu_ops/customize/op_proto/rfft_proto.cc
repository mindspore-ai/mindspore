/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
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

#include "custom_op_proto/cust_math_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"
#include "utils/op_common_util.h"
#include "utils/op_const.h"

namespace ge {
static graphStatus RFFTInferShapeCommon(Operator &op, int64_t n, int64_t dim, bool unknown_n) {
  if (!unknown_n && n <= 0) {
    std::string err_msg = GetAttrValueErrMsg("rfft n", std::to_string(n), ConcatString("n > 0"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  const int kRealFFTSideNum = 2;
  auto input_desc = op.GetInputDescByName("input");
  auto out_desc = op.GetOutputDescByName("y");

  DataType x_dtype = input_desc.GetDataType();
  DataType y_dtype;
  if (x_dtype == DT_DOUBLE) {
    y_dtype = DT_COMPLEX128;
  } else {
    y_dtype = DT_COMPLEX64;
  }
  out_desc.SetDataType(y_dtype);

  bool unknown_rank_shape = IsUnknownRankShape(input_desc.GetShape());
  if (unknown_rank_shape) {
    out_desc.SetShape(ge::Shape(UNKNOWN_RANK));
    OP_LOGD(TbeGetName(op).c_str(), "output shape:%s", to_string(out_desc.GetShape()).c_str());
    op.UpdateOutputDesc("y", out_desc);
    return GRAPH_SUCCESS;
  }

  size_t x_rank = input_desc.GetShape().GetDimNum();
  auto input_shape_dims = input_desc.GetShape().GetDims();
  dim = dim < 0 ? static_cast<int64_t>(x_rank) + dim : dim;
  vector<int64_t> output_shape_dims(input_shape_dims.begin(), input_shape_dims.end());
  if (unknown_n) {
    if (input_shape_dims[dim] != UNKNOWN_DIM) {
      output_shape_dims[dim] = output_shape_dims[dim] / kRealFFTSideNum + 1;
    }
  } else {
    output_shape_dims[dim] = n / kRealFFTSideNum + 1;
  }

  out_desc.SetShape(ge::Shape(output_shape_dims));
  OP_LOGD(TbeGetName(op).c_str(), "output shape:%s", to_string(out_desc.GetShape()).c_str());
  op.UpdateOutputDesc("y", out_desc);

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(RFFTInferShape) {
  const vector<string> depend_names = {"n", "dim"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  // infer output shape based on 'n' and 'dim'
  Tensor n_data;
  bool is_unknown_n{true};
  if (op.GetInputConstData("n", n_data) == GRAPH_SUCCESS) {
    is_unknown_n = false;
  }
  OP_LOGD(TbeGetName(op), "rfft n is unknown[%s].", is_unknown_n ? "true" : "false");
  int64_t n = 0;
  if (!is_unknown_n) {
    DataType dtype = op.GetInputDescByName("n").GetDataType();
    std::vector<int64_t> const_vec;
    if (!GetConstValue(op, n_data, dtype, const_vec)) {
      is_unknown_n = true;
      OP_LOGW(TbeGetName(op), "Get rfft n value failed.");
    } else {
      n = const_vec[0];
    }
  }
  Tensor dim_data;
  bool is_unknown_axis{true};
  if (op.GetInputConstData("dim", dim_data) == GRAPH_SUCCESS) {
    is_unknown_axis = false;
  }
  OP_LOGD(TbeGetName(op), "rfft axis is unknown[%s].", is_unknown_axis ? "true" : "false");
  int64_t dim = -1;
  if (!is_unknown_axis) {
    DataType dim_dtype = op.GetInputDescByName("dim").GetDataType();
    std::vector<int64_t> const_vec_dim;
    if (!GetConstValue(op, dim_data, dim_dtype, const_vec_dim)) {
      OP_LOGW(TbeGetName(op), "Get rfft dim value failed.");
    } else {
      dim = const_vec_dim[0];
    }
  }

  return RFFTInferShapeCommon(op, n, dim, is_unknown_n);
}
CUST_COMMON_INFER_FUNC_REG(RFFT, RFFTInferShape);
}  // namespace ge