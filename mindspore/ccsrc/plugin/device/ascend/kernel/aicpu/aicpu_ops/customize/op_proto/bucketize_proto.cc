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

namespace ge {
// ---------Bucketize------------------
IMPLEMT_INFERFUNC(Bucketize, BucketizeInfer) {
  std::string str_name = TbeGetName(op);
  const char *opname = str_name.c_str();
  OP_LOGD(opname, "Enter Bucketize inferfunction!");

  // set output shape
  std::vector<int64_t> x_shape = op.GetInputDesc("x").GetShape().GetDims();
  auto output_desc = op.GetOutputDesc(0);
  output_desc.SetShape(Shape(x_shape));

  // set output dtype
  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("dtype");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("get attr[dtype] failed"));
    return GRAPH_FAILED;
  }
  if ((dtype != DT_INT32) && (dtype != DT_INT64)) {
    std::string err_msg = GetInputInvalidErrMsg("dtype");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("The attr [dtype] must be one of DT_INT32 or DT_INT64"));
    return GRAPH_FAILED;
  }
  output_desc.SetDataType(dtype);
  op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Bucketize, BucketizeVerify) { return GRAPH_SUCCESS; }
INFER_FUNC_REG(Bucketize, BucketizeInfer);
VERIFY_FUNC_REG(Bucketize, BucketizeVerify);
// ---------Bucketize End-------------------
}  // namespace ge