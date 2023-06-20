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

#include "inc/bessel_i0_op.h"
#include <vector>
#include <string>
namespace ge {

IMPLEMT_COMMON_INFERFUNC(BesselI0InferShape) {
  TensorDesc tensordesc_output = op.GetOutputDescByName("y");
  tensordesc_output.SetShape(op.GetInputDescByName("x").GetShape());
  tensordesc_output.SetDataType(op.GetInputDescByName("x").GetDataType());
  tensordesc_output.SetFormat(op.GetInputDescByName("x").GetFormat());
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_VERIFIER(BesselI0, BesselI0Verify) { return GRAPH_SUCCESS; }

CUST_COMMON_INFER_FUNC_REG(BesselI0, BesselI0InferShape);

CUST_VERIFY_FUNC_REG(BesselI0, BesselI0Verify);

}  // namespace ge
