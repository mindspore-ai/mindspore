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

#include "src/ops/populate/populate_register.h"
#include "nnacl/random_parameter.h"
using mindspore::schema::PrimitiveType_RandomStandardNormal;

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateRandomStandardNormalParameter(const void *prim) {
  auto *primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_RandomStandardNormal();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<RandomParam *>(malloc(sizeof(RandomParam)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc RandomParam failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(RandomParam));

  param->op_parameter_.type_ = primitive->value_type();
  param->seed_ = value->seed();
  param->seed2_ = value->seed2();
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

REG_POPULATE(PrimitiveType_RandomStandardNormal, PopulateRandomStandardNormalParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
