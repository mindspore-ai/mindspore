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

#ifndef PREDICT_CONVERTER_COMMON_OP_UTILS_H_
#define PREDICT_CONVERTER_COMMON_OP_UTILS_H_

#include <functional>
#include <string>
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
inline schema::PrimitiveType GetCNodeTType(const schema::CNodeT &cNodeT) { return cNodeT.primitive->value.type; }
inline std::string GetCNodeTTypeName(const schema::CNodeT &cNodeT) {
  return schema::EnumNamePrimitiveType(GetCNodeTType(cNodeT));
}
}  // namespace lite
}  // namespace mindspore

#endif  // PREDICT_CONVERTER_COMMON_OP_UTILS_H_

