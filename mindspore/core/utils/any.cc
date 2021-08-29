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

#include "utils/any.h"
#ifndef _MSC_VER
#include <cxxabi.h>
#endif
#include <memory>

namespace mindspore {
// only support (int, float, bool) as Literal
bool AnyIsLiteral(const Any &any) {
  static const std::type_index typeid_int = std::type_index(typeid(int));
  static const std::type_index typeid_float = std::type_index(typeid(float));
  static const std::type_index typeid_bool = std::type_index(typeid(bool));

  auto typeid_any = std::type_index(any.type());
  return typeid_int == typeid_any || typeid_float == typeid_any || typeid_bool == typeid_any;
}

Any &Any::operator=(const Any &other) {
  if (m_ptr == other.m_ptr || &other == this) {
    return *this;
  }
  m_ptr = other.clone();
  m_tpIndex = other.m_tpIndex;
  return *this;
}

bool Any::operator<(const Any &other) const { return this < &other; }

Any &Any::operator=(Any &&other) {
  if (this != &other) {
    if (m_ptr == other.m_ptr || &other == this) {
      return *this;
    }

    m_ptr = std::move(other.m_ptr);
    m_tpIndex = std::move(other.m_tpIndex);
    other.m_ptr = nullptr;
  }
  return *this;
}
}  // namespace mindspore
