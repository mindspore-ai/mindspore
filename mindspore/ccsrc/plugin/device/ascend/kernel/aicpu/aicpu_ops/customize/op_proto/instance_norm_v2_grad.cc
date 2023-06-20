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

#include "inc/instance_norm_v2_grad.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"
namespace ge {
// ----------------InstanceNormV2Grad Begin-------------------
IMPLEMT_COMMON_INFERFUNC(InstanceNormV2GradInferShape) {
  // x desc and gamma desc
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto dy_desc = op_info->MutableInputDesc("dy");
  auto dy_shape = dy_desc->MutableShape();
  auto dy_dtype = dy_desc->GetDataType();
  auto gamma_desc = op_info->MutableInputDesc("gamma");
  auto gamma_shape = gamma_desc->MutableShape();
  auto gamma_dtype = gamma_desc->GetDataType();

  // update output desc
  auto pd_x_desc = op_info->MutableOutputDesc("pd_x");
  auto pd_gamma_desc = op_info->MutableOutputDesc("pd_gamma");
  auto pd_beta_desc = op_info->MutableOutputDesc("pd_beta");
  pd_x_desc->SetShape(dy_shape);
  pd_gamma_desc->SetShape(gamma_shape);
  pd_beta_desc->SetShape(gamma_shape);
  pd_x_desc->SetDataType(dy_dtype);
  pd_gamma_desc->SetDataType(gamma_dtype);
  pd_beta_desc->SetDataType(gamma_dtype);

  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(InstanceNormV2Grad, InstanceNormV2GradInferShape);
// ----------------InstanceNormV2Grad END---------------------
}  // namespace ge