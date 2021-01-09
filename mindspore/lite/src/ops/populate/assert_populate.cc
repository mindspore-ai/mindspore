/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/assert_op.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateAssertParameter(const mindspore::lite::PrimitiveC *primitive) {
  OpParameter *assert_parameter = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  if (assert_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc AssertParameter failed.";
    return nullptr;
  }
  memset(assert_parameter, 0, sizeof(OpParameter));
  assert_parameter->type_ = primitive->Type();

  return reinterpret_cast<OpParameter *>(assert_parameter);
}
Registry AssertParameterRegistry(schema::PrimitiveType_Assert, PopulateAssertParameter);
}  // namespace lite
}  // namespace mindspore
