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
#include "nnacl/power_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulatePowerParameter(const void *prim) {
  PowerParameter *power_param = reinterpret_cast<PowerParameter *>(malloc(sizeof(PowerParameter)));
  if (power_param == nullptr) {
    MS_LOG(ERROR) << "malloc PowerParameter failed.";
    return nullptr;
  }
  memset(power_param, 0, sizeof(PowerParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  power_param->op_parameter_.type_ = primitive->value_type();
  auto power_prim = primitive->value_as_PowFusion();
  power_param->scale_ = power_prim->scale();
  power_param->shift_ = power_prim->shift();
  return reinterpret_cast<OpParameter *>(power_param);
}
}  // namespace

Registry g_powerParameterRegistry(schema::PrimitiveType_PowFusion, PopulatePowerParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
