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
#include "src/common/ops/populate/populate_register.h"
#include "nnacl/flatten_parameter.h"
using mindspore::schema::PrimitiveType_Flatten;

namespace mindspore {
namespace lite {
OpParameter *PopulateFlattenParameter(const void *prim) {
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_Flatten();
  if (value == nullptr) {
    MS_LOG(ERROR) << "param is nullptr";
    return nullptr;
  }
  auto *param = reinterpret_cast<FlattenParameter *>(malloc(sizeof(FlattenParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc FlattenParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(FlattenParameter));
  param->axis_ = static_cast<int>(value->axis());
  param->op_parameter_.type_ = static_cast<int>(primitive->value_type());
  return reinterpret_cast<OpParameter *>(param);
}
REG_POPULATE(PrimitiveType_Flatten, PopulateFlattenParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
