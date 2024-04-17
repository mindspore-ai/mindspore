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

#include "op_proto/inc/linalg_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"
#include "utils/linalg_ops_shape_fns.h"

namespace ge {
IMPLEMT_INFERFUNC(Cholesky, CholeskyInfer) {
  auto x_desc = op.GetInputDesc(0);

  Shape y_shape;
  if (MakeBatchSquareMatrix(x_desc, y_shape, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Op Cholesky first input x's tensor make batch square matrix failed.");
    return GRAPH_FAILED;
  }
  DataType type = x_desc.GetDataType();

  auto y_desc = op.GetOutputDesc(0);
  y_desc.SetShape(y_shape);
  y_desc.SetDataType(type);
  op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Cholesky, CholeskyInfer);
}  // namespace ge