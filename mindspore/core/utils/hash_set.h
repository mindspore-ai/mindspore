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

#ifndef MINDSPORE_CORE_UTILS_HASH_SET_H_
#define MINDSPORE_CORE_UTILS_HASH_SET_H_

#include <functional>
#if (ENABLE_FAST_HASH_TABLE) && __has_include("include/robin_hood.h")
#include "include/robin_hood.h"

namespace mindspore {
template <typename T, typename Hash = robin_hood::hash<T>, typename Equal = std::equal_to<T>>
using HashSet = robin_hood::unordered_set<T, Hash, Equal>;

#else
#include <unordered_set>

namespace mindspore {
template <typename T, typename Hash = std::hash<T>, typename Equal = std::equal_to<T>>
using HashSet = std::unordered_set<T, Hash, Equal>;

#endif
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_HASH_SET_H_
