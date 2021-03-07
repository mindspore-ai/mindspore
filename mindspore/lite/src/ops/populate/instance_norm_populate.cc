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
#include "nnacl/instance_norm_parameter.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateInstanceNormParameter(const void *prim) {
  InstanceNormParameter *instance_norm_param =
    reinterpret_cast<InstanceNormParameter *>(malloc(sizeof(InstanceNormParameter)));
  if (instance_norm_param == nullptr) {
    MS_LOG(ERROR) << "malloc InstanceNormParameter failed.";
    return nullptr;
  }
  memset(instance_norm_param, 0, sizeof(InstanceNormParameter));

  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_InstanceNorm();
  instance_norm_param->op_parameter_.type_ = primitive->value_type();
  instance_norm_param->epsilon_ = value->epsilon();
  return reinterpret_cast<OpParameter *>(instance_norm_param);
}

Registry InstanceNormParameterRegistry(schema::PrimitiveType_InstanceNorm, PopulateInstanceNormParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
