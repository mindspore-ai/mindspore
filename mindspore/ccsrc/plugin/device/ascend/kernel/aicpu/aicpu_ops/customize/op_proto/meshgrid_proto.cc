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

#include "custom_op_proto/cust_array_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
// ---------------Meshgrid-------------------
IMPLEMT_COMMON_INFERFUNC(MeshgridInfer) {
  auto input_size = op.GetInputsSize();
  if (input_size == 0) {
    return GRAPH_SUCCESS;
  }
  std::vector<int64_t> out_dims = std::vector<int64_t>();
  auto out_dtype = op.GetDynamicInputDesc("x", 0).GetDataType();
  for (size_t i = 0; i < op.GetInputsSize(); ++i) {
    auto in_shape = op.GetInputDesc(i).GetShape();
    if (in_shape.GetDims() == UNKNOWN_RANK) {
      out_dims = ge::UNKNOWN_RANK;
      break;
    }
    out_dims.push_back(in_shape.GetDim(0));
  }
  std::string indexing;
  if (op.GetAttr("indexing", indexing) == GRAPH_SUCCESS) {
    if (indexing == "xy") {
      std::swap(out_dims[0], out_dims[1]);
    }
  } else {
    OP_LOGW(TbeGetName(op).c_str(), "get attr indexing failed.");
  }
  Shape out_shape = Shape(out_dims);
  for (size_t i = 0; i < op.GetInputsSize(); ++i) {
    TensorDesc output_desc = op.GetDynamicOutputDesc("y", i);
    output_desc.SetShape(out_shape);
    output_desc.SetDataType(out_dtype);
    op.UpdateDynamicOutputDesc("y", i, output_desc);
  }
  return GRAPH_SUCCESS;
}
CUST_INFER_FUNC_REG(Meshgrid, MeshgridInfer);
// ---------------Meshgrid End---------------
}  // namespace ge