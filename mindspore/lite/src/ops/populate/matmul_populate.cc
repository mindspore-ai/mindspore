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

OpParameter *PopulateMatMulParameter(const void *prim) {
  MatMulParameter *matmul_param = reinterpret_cast<MatMulParameter *>(malloc(sizeof(MatMulParameter)));
  if (matmul_param == nullptr) {
    MS_LOG(ERROR) << "malloc MatMulParameter failed.";
    return nullptr;
  }
  memset(matmul_param, 0, sizeof(MatMulParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_MatMul();
  matmul_param->op_parameter_.type_ = primitive->value_type();
  matmul_param->b_transpose_ = value->transpose_b();
  matmul_param->a_transpose_ = value->transpose_a();
  matmul_param->has_bias_ = false;
  matmul_param->act_type_ = ActType_No;
  return reinterpret_cast<OpParameter *>(matmul_param);
}
Registry MatMulParameterRegistry(schema::PrimitiveType_MatMul, PopulateMatMulParameter, SCHEMA_CUR);

}  // namespace lite
}  // namespace mindspore
