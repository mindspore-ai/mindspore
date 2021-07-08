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
#include "nnacl/fp32/activation_fp32.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateGeLUParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto gelu_prim = primitive->value_as_GeLU();
  if (gelu_prim == nullptr) {
    MS_LOG(ERROR) << "activation_prim is nullptr";
    return nullptr;
  }
  auto *act_param = reinterpret_cast<ActivationParameter *>(malloc(sizeof(ActivationParameter)));
  if (act_param == nullptr) {
    MS_LOG(ERROR) << "malloc ActivationParameter failed.";
    return nullptr;
  }
  memset(act_param, 0, sizeof(ActivationParameter));
  act_param->op_parameter_.type_ = schema::PrimitiveType_Activation;
  act_param->type_ = static_cast<int>(schema::ActivationType_GELU);
  return reinterpret_cast<OpParameter *>(act_param);
}
}  // namespace

Registry g_GeLUV0ParameterRegistry(schema::v0::PrimitiveType_GeLU, PopulateGeLUParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
