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
#include "nnacl/fp32_grad/activation_grad.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateActivationGradParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto activation_grad_prim = primitive->value_as_ActivationGrad();
  ActivationGradParameter *act_param =
    reinterpret_cast<ActivationGradParameter *>(malloc(sizeof(ActivationGradParameter)));
  if (act_param == nullptr) {
    MS_LOG(ERROR) << "malloc ActivationParameter failed.";
    return nullptr;
  }
  memset(act_param, 0, sizeof(ActivationGradParameter));
  act_param->op_parameter.type_ = schema::PrimitiveType_ActivationGrad;

  act_param->type_ = static_cast<int>(activation_grad_prim->type());
  act_param->alpha_ = activation_grad_prim->alpha();
  return reinterpret_cast<OpParameter *>(act_param);
}
}  // namespace

Registry g_activationGradV0ParameterRegistry(schema::v0::PrimitiveType_ActivationGrad, PopulateActivationGradParameter,
                                             SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
