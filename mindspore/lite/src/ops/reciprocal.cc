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

#include "src/ops/reciprocal.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifndef PRIMITIVE_WRITEABLE
PrimitiveC *ReciprocalCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<Reciprocal>(primitive);
}
Registry ReciprocalRegistry(schema::PrimitiveType_Reciprocal, ReciprocalCreator);
#endif
}  // namespace lite
}  // namespace mindspore
