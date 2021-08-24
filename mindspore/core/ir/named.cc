/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "ir/named.h"
#include "abstract/abstract_value.h"

namespace mindspore {
bool Named::operator==(const Value &other) const {
  if (other.isa<Named>()) {
    auto other_named = static_cast<const Named &>(other);
    return *this == other_named;
  } else {
    return false;
  }
}

abstract::AbstractBasePtr None::ToAbstract() { return std::make_shared<abstract::AbstractNone>(); }

abstract::AbstractBasePtr Null::ToAbstract() { return std::make_shared<abstract::AbstractNull>(); }

abstract::AbstractBasePtr Ellipsis::ToAbstract() { return std::make_shared<abstract::AbstractEllipsis>(); }
}  // namespace mindspore
