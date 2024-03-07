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

namespace ge {
IMPLEMT_COMMON_INFERFUNC(FFTNBaseInferShape) {
  auto input_desc = op.GetInputDescByName("input");
  auto out_desc = op.GetOutputDescByName("y");

  DataType x_dtype = input_desc.GetDataType();
  DataType y_dtype;
  if (x_dtype == DT_DOUBLE || x_dtype == DT_COMPLEX128) {
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
  vector<int64_t> output_shape_dims(input_shape_dims.begin(), input_shape_dims.end());
  const vector<string> depend_names = {"n", "dim"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  // infer output shape based on 'n' and 'dim'
  Tensor s_tensor;
  if (op.GetInputConstData("s", s_tensor) == GRAPH_SUCCESS) {
    DataType dtype = op.GetInputDescByName("s").GetDataType();
    std::vector<int64_t> s_vec;
    GetConstValue(op, s_tensor, dtype, s_vec);

    Tensor dim_tensor;
    std::vector<int64_t> dim_vec;
    if (op.GetInputConstData("dim", dim_tensor) == GRAPH_SUCCESS) {
      DataType dim_dtype = op.GetInputDescByName("dim").GetDataType();
      GetConstValue(op, dim_tensor, dim_dtype, dim_vec);
      for (size_t i = 0; i < dim_vec.size(); i++) {
        dim_vec[i] = dim_vec[i] < 0 ? static_cast<int64_t>(x_rank) + dim_vec[i] : dim_vec[i];
      }
    } else {
      for (size_t i = 0; i < s_vec.size(); i++) {
        (void)dim_vec.emplace_back(x_rank - s_vec.size() + i);
      }
    }
    for (size_t i = 0; i < s_vec.size(); i++) {
      output_shape_dims[dim_vec[i]] = s_vec[i];
    }
  }

  out_desc.SetShape(ge::Shape(output_shape_dims));
  op.UpdateOutputDesc("y", out_desc);
  return GRAPH_SUCCESS;
}
CUST_COMMON_INFER_FUNC_REG(FFT2, FFTNBaseInferShape);
CUST_COMMON_INFER_FUNC_REG(FFTN, FFTNBaseInferShape);
CUST_COMMON_INFER_FUNC_REG(IFFT2, FFTNBaseInferShape);
CUST_COMMON_INFER_FUNC_REG(IFFTN, FFTNBaseInferShape);
}  // namespace ge
