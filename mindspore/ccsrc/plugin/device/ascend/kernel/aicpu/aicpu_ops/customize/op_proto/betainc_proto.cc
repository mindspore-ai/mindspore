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

#include "op_proto/inc/math_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/op_const.h"
#include "utils/common_shape_fns.h"

namespace ge {
IMPLEMT_INFERFUNC(Betainc, BetaincInfer) {
  const int num_inputs = 3;
  Shape output(UNKNOWN_RANK);
  int num_scalars = 0;
  Shape some_non_scalar;
  for (int i = 0; i < num_inputs; ++i) {
    TensorDesc input_desc = op.GetInputDesc(i);
    Shape input_shape = input_desc.GetShape();
    if (!RankKnown(input_shape)) {
      some_non_scalar = input_shape;
    } else if (input_shape.GetDimNum() == 0) {
      ++num_scalars;
    } else {
      if (Merge(output, input_shape, output, op) != GRAPH_SUCCESS) {
        std::string err_msg =
          ConcatString("failed to call Merge function to merge", i, "th input shape",
                       DebugString(input_shape.GetDims()), " and output[z] shape", DebugString(output.GetDims()));
        AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
      }
      some_non_scalar = output;
    }
  }

  if (num_scalars == num_inputs - 1) {
    output = some_non_scalar;
  } else if (num_scalars == num_inputs) {
    TensorDesc a_desc = op.GetInputDescByName("a");
    output = a_desc.GetShape();
  }
  DataType a_type = op.GetInputDescByName("a").GetDataType();
  TensorDesc z_desc = op.GetOutputDescByName("z");
  z_desc.SetShape(output);
  z_desc.SetDataType(a_type);
  if (op.UpdateOutputDesc("z", z_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("fail to update output[z] desc."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Betainc, BetaincInfer);
}  // namespace ge