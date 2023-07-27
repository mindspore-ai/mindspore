/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_COMMON_OPS_OPERATOR_POPULATE_UTILS_H_
#define MINDSPORE_LITE_SRC_COMMON_OPS_OPERATOR_POPULATE_UTILS_H_

#include <string>
#include "ops/base_operator.h"

namespace mindspore::lite {
template <typename T>
T GetAttrWithDefault(const BaseOperatorPtr &base_operator, const std::string &key, T default_value) {
  if (MS_UNLIKELY(base_operator == nullptr)) {
    return default_value;
  }
  auto prim = base_operator->GetPrim();
  if (MS_UNLIKELY(prim == nullptr)) {
    return default_value;
  }
  auto attr = prim->GetAttr(key);
  if (attr == nullptr) {
    return default_value;
  }
  return GetValue<T>(attr);
}
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_COMMON_OPS_OPERATOR_POPULATE_UTILS_H_
