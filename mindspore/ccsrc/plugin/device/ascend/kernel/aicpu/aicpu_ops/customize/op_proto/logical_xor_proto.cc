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

#include "inc/logical_xor_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
CUST_IMPLEMT_VERIFIER(LogicalXor, LogicalXorVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(LogicalXorInferShape) {
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto vec_y = op_desc->MutableOutputDesc("y")->MutableShape().GetDims();
  if (IsUnknownRankShape(vec_y) || IsUnknownVec(vec_y)) {
    if (!InferShapeRangeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
      return GRAPH_FAILED;
    }
  }

  op_desc->MutableOutputDesc("y")->SetDataType(DT_BOOL);
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(LogicalXor, LogicalXorInferShape);
CUST_VERIFY_FUNC_REG(LogicalXor, LogicalXorVerify);
}  // namespace ge