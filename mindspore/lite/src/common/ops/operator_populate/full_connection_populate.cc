/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl/matmul_parameter.h"
#include "ops/fusion/full_connection.h"
using mindspore::ops::kNameFullConnection;
using mindspore::schema::PrimitiveType_FullConnection;
namespace mindspore {
namespace lite {
OpParameter *PopulateFullconnectionOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<MatMulParameter *>(PopulateOpParameter<MatMulParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new MatMulParameter failed.";
    return nullptr;
  }
  param->b_transpose_ = true;
  param->a_transpose_ = false;

  auto op = dynamic_cast<ops::FullConnection *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "base_operator cast to FullConnection failed";
    free(param);
    return nullptr;
  }
  param->has_bias_ = op->get_has_bias();
  auto act_type = static_cast<ActType>(op->get_activation_type());
  if (act_type == ActType_Relu || act_type == ActType_Relu6) {
    param->act_type_ = act_type;
  } else {
    param->act_type_ = ActType_No;
  }
  auto axis = static_cast<int>(op->get_axis());
  CHECK_LESS_RETURN_RET(INT32_MAX, axis, nullptr, param);
  param->axis_ = axis;
  param->use_axis_ = op->get_use_axis();
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameFullConnection, PrimitiveType_FullConnection, PopulateFullconnectionOpParameter)
}  // namespace lite
}  // namespace mindspore
