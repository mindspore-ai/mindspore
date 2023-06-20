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

#include "inc/lstsq_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
IMPLEMT_COMMON_INFERFUNC(LstsqInferShape) {
  Shape shape_1 = op.GetInputDescByName("matrix").GetShape();
  Shape shape_2 = op.GetInputDescByName("rhs").GetShape();
  DataType input_dtype = op.GetInputDescByName("matrix").GetDataType();
  Format input_format = op.GetInputDescByName("matrix").GetFormat();

  std::vector<int64_t> dim_vector;
  if (shape_2.GetDimNum() == 1) {
    dim_vector.push_back(shape_1.GetDim(1));
    dim_vector.push_back(1);
  } else {
    dim_vector.push_back(shape_1.GetDim(1));
    dim_vector.push_back(shape_2.GetDim(1));
  }
  Shape output_shape(dim_vector);
  TensorDesc td = op.GetOutputDescByName("y");
  td.SetShape(output_shape);
  td.SetDataType(input_dtype);
  td.SetFormat(input_format);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_VERIFIER(Lstsq, LstsqVerify) { return GRAPH_SUCCESS; }
CUST_COMMON_INFER_FUNC_REG(Lstsq, LstsqInferShape);
CUST_VERIFY_FUNC_REG(Lstsq, LstsqVerify);
}  // namespace ge