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

#include "inc/glu_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
// ----------------Glu Begin------------------------
IMPLEMT_COMMON_INFERFUNC(GluInferShape) {
  auto opname = TbeGetName(op).c_str();
  Shape x_shape = op.GetInputDescByName("x").GetShape();
  DataType x_dtype = op.GetInputDescByName("x").GetDataType();
  Format x_format = op.GetInputDescByName("x").GetFormat();

  int64_t split_dim;
  if (op.GetAttr("axis", split_dim) == GRAPH_FAILED) {
    split_dim = -1;
  }

  int64_t dim_num = static_cast<int64_t>(x_shape.GetDimNum());
  if ((split_dim < -dim_num) || (split_dim >= dim_num)) {
    OP_LOGE(opname, "The value of the attribute axis is out of range.");
    return GRAPH_FAILED;
  }
  if (split_dim < 0) {
    split_dim += dim_num;
  }
  int64_t split_dim_value = x_shape.GetDim(split_dim);
  if (split_dim_value % 2 != 0) {
    OP_LOGE(opname, "The value of thea attribute axis is not even\n");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> dim_vector = x_shape.GetDims();
  dim_vector[split_dim] = split_dim_value / 2;

  Shape output_shape(dim_vector);
  TensorDesc td = op.GetOutputDescByName("y");
  td.SetShape(output_shape);
  td.SetDataType(x_dtype);
  td.SetFormat(x_format);
  op.UpdateOutputDesc("y", td);

  return GRAPH_SUCCESS;
}
CUST_COMMON_INFER_FUNC_REG(Glu, GluInferShape);
// ----------------Glu END---------------------------------
}  // namespace ge