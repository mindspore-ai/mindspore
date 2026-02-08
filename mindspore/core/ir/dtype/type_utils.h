/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_DTYPE_TYPE_UTILS_H_
#define MINDSPORE_CORE_IR_DTYPE_TYPE_UTILS_H_

#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include <type_traits>
#include <algorithm>

#include "utils/hash_map.h"
#include "base/base.h"
#include "ir/named.h"
#include "ir/dtype/type_id.h"

namespace mindspore {
TypeId IntBitsToTypeId(const int nbits);
TypeId UIntBitsToTypeId(const int nbits);
TypeId FloatBitsToTypeId(const int nbits);
TypeId BFloatBitsToTypeId(const int nbits);
TypeId ComplexBitsToTypeId(const int nbits);

bool IsSameObjectType(const Type &lhs, const Type &rhs);
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_DTYPE_TYPE_UTILS_H_
