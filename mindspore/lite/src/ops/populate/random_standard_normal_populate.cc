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

#include "src/ops/random_standard_normal.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/random_standard_normal_parameter.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateRandomStandardNormalParameter(const mindspore::lite::PrimitiveC *primitive) {
  RandomStandardNormalParam *random_parameter =
    reinterpret_cast<RandomStandardNormalParam *>(malloc(sizeof(RandomStandardNormalParam)));
  if (random_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc RandomStandardNormal parameter failed.";
    return nullptr;
  }
  memset(random_parameter, 0, sizeof(RandomStandardNormalParam));
  random_parameter->op_parameter_.type_ = primitive->Type();
  auto param =
    reinterpret_cast<mindspore::lite::RandomStandardNormal *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  random_parameter->seed_ = param->GetSeed();
  random_parameter->seed2_ = param->GetSeed2();
  return reinterpret_cast<OpParameter *>(random_parameter);
}
Registry RandomStandardNormalParameterRegistry(schema::PrimitiveType_RandomStandardNormal,
                                               PopulateRandomStandardNormalParameter);
}  // namespace lite
}  // namespace mindspore
