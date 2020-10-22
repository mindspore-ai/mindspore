/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/flatten.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateFlattenParameter(const mindspore::lite::PrimitiveC *primitive) {
  FlattenParameter *flatten_param = reinterpret_cast<FlattenParameter *>(malloc(sizeof(FlattenParameter)));
  if (flatten_param == nullptr) {
    MS_LOG(ERROR) << "malloc FlattenParameter failed.";
    return nullptr;
  }
  memset(flatten_param, 0, sizeof(FlattenParameter));
  flatten_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(flatten_param);
}

Registry FlattenParameterRegistry(schema::PrimitiveType_Flatten, PopulateFlattenParameter);

}  // namespace lite
}  // namespace mindspore
