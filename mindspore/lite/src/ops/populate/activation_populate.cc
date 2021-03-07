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
#include "nnacl/fp32/activation_fp32.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateRelu6Parameter(const void *prim) {
  ActivationParameter *act_param = reinterpret_cast<ActivationParameter *>(malloc(sizeof(ActivationParameter)));
  if (act_param == nullptr) {
    MS_LOG(ERROR) << "malloc ActivationParameter failed.";
    return nullptr;
  }
  memset(act_param, 0, sizeof(ActivationParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  act_param->op_parameter_.type_ = primitive->value_type();
  auto acti_prim = primitive->value_as_Activation();
  act_param->type_ = static_cast<int>(acti_prim->activation_type());
  act_param->alpha_ = acti_prim->alpha();
  act_param->min_val_ = acti_prim->min_val();
  act_param->max_val_ = acti_prim->max_val();
  return reinterpret_cast<OpParameter *>(act_param);
}
}  // namespace

Registry g_relu6ParameterRegistry(schema::PrimitiveType_Activation, PopulateRelu6Parameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
