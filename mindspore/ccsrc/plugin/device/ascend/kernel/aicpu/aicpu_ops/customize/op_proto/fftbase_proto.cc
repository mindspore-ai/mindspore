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
IMPLEMT_COMMON_INFERFUNC(FFTBaseInferShape) {
  DataType x_dtype = op.GetInputDescByName("x").GetDataType();
  DataType y_dtype;
  if (x_dtype == DT_DOUBLE || x_dtype == DT_COMPLEX128) {
    y_dtype = DT_COMPLEX128;
  } else {
    y_dtype = DT_COMPLEX64;
  }
  TensorDesc out_desc = op.GetOutputDescByName("y");
  out_desc.SetDataType(y_dtype);

  vector<int64_t> s;
  op.GetAttr("s", s);
  // If s is given, the input will be zero-padded or trimmed to this length.
  if (s.size() == 0) {
    out_desc.SetShape(op.GetInputDescByName("x").GetShape());
  } else {
    vector<int64_t> x_shape_list = op.GetInputDescByName("x").GetShape().GetDims();
    std::vector<int64_t> out_shape_list = {};
    out_shape_list.assign(x_shape_list.begin(), x_shape_list.end());

    int64_t x_rank = x_shape_list.size();
    vector<int64_t> dims;
    op.GetAttr("dims", dims);
    int64_t tmp_pos;
    for (size_t i = 0; i < s.size(); i++) {
      if (dims.size() == 0) {
        tmp_pos = x_rank - s.size() + i;
        out_shape_list[tmp_pos] = s[i];
      } else {
        tmp_pos = dims[i] < 0 ? x_rank + dims[i] : dims[i];
        out_shape_list[tmp_pos] = s[i];
      }
    }
    Shape out_shape(out_shape_list);
    out_desc.SetShape(out_shape);
  }

  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Failed to update output desc.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
CUST_COMMON_INFER_FUNC_REG(FFTBase, FFTBaseInferShape);
}  // namespace ge