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

#include "inc/matrix_logarithm.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"
namespace ge {
// ----------------MatrixLogarithm--------------------
IMPLEMT_COMMON_INFERFUNC(MatrixLogarithmInferShaper) {
  auto x_shape = op.GetInputDescByName("x").GetShape().GetDims();
  Shape input_shape = op.GetInputDescByName("x").GetShape();
  DataType input_dtype = op.GetInputDescByName("x").GetDataType();
  int64_t size_num = op.GetInputDescByName("x").GetShape().GetDimNum();
  TensorDesc td = op.GetOutputDescByName("y");
  td.SetShape(ge::Shape(input_shape));
  td.SetDataType(input_dtype);
  if (op.UpdateOutputDesc("y", td) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  if (size_num < 2) {
    string err_msg = ConcatString("the input[x] should be greater than 2, but get ", size_num, ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
// 注册函数
CUST_COMMON_INFER_FUNC_REG(MatrixLogarithm, MatrixLogarithmInferShaper);
// ----------------MatrixLogarithm END-------------------
}  // namespace ge