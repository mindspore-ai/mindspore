/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "schema/model_v0_generated.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/matmul_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateMatMulParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto matmul_prim = primitive->value_as_MatMul();

  MatMulParameter *matmul_param = reinterpret_cast<MatMulParameter *>(malloc(sizeof(MatMulParameter)));
  if (matmul_param == nullptr) {
    MS_LOG(ERROR) << "malloc MatMulParameter failed.";
    return nullptr;
  }
  memset(matmul_param, 0, sizeof(MatMulParameter));
  matmul_param->op_parameter_.type_ = schema::PrimitiveType_MatMul;
  matmul_param->b_transpose_ = matmul_prim->transposeB();
  matmul_param->a_transpose_ = matmul_prim->transposeA();
  matmul_param->has_bias_ = false;
  matmul_param->act_type_ = ActType_No;

  return reinterpret_cast<OpParameter *>(matmul_param);
}
}  // namespace

Registry g_MatMulPV0arameterRegistry(schema::v0::PrimitiveType_MatMul, PopulateMatMulParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
