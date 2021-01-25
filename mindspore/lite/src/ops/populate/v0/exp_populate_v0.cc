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
#include "nnacl/fp32/exp_fp32.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateExpParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto exp_prim = primitive->value_as_Exp();
  ExpParameter *exp_parameter = reinterpret_cast<ExpParameter *>(malloc(sizeof(ExpParameter)));
  if (exp_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc ExpParameter failed.";
    return nullptr;
  }
  memset(exp_parameter, 0, sizeof(ExpParameter));
  exp_parameter->op_parameter_.type_ = schema::PrimitiveType_ExpFusion;

  exp_parameter->base_ = exp_prim->base();
  exp_parameter->scale_ = exp_prim->scale();
  exp_parameter->shift_ = exp_prim->shift();
  if (exp_parameter->base_ != -1 && exp_parameter->base_ <= 0) {
    MS_LOG(ERROR) << "Exp base must be strictly positive, got " << exp_parameter->base_;
    free(exp_parameter);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(exp_parameter);
}
}  // namespace

Registry g_expV0ParameterRegistry(schema::v0::PrimitiveType_Exp, PopulateExpParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
