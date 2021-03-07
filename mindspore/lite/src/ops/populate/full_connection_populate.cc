/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "src/ops/populate/populate_register.h"
#include "nnacl/matmul_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateFullconnectionParameter(const void *prim) {
  MatMulParameter *matmul_param = reinterpret_cast<MatMulParameter *>(malloc(sizeof(MatMulParameter)));
  if (matmul_param == nullptr) {
    MS_LOG(ERROR) << "malloc MatMulParameter failed.";
    return nullptr;
  }
  memset(matmul_param, 0, sizeof(MatMulParameter));
  auto *primitive = static_cast<const schema::Primitive *>(prim);
  matmul_param->op_parameter_.type_ = primitive->value_type();
  auto full_conn_prim = primitive->value_as_FullConnection();
  matmul_param->b_transpose_ = true;
  matmul_param->a_transpose_ = false;
  matmul_param->has_bias_ = full_conn_prim->has_bias();
  if (full_conn_prim->activation_type() == schema::ActivationType_RELU) {
    matmul_param->act_type_ = ActType_Relu;
  } else if (full_conn_prim->activation_type() == schema::ActivationType_RELU6) {
    matmul_param->act_type_ = ActType_Relu6;
  } else {
    matmul_param->act_type_ = ActType_No;
  }
  matmul_param->axis_ = full_conn_prim->axis();
  matmul_param->use_axis_ = full_conn_prim->use_axis();
  return reinterpret_cast<OpParameter *>(matmul_param);
}
}  // namespace

Registry g_fullConnRegistry(schema::PrimitiveType_FullConnection, PopulateFullconnectionParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
