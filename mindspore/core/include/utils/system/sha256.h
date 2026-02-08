/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_UTILS_SYSTEM_SHA256_H_
#define MINDSPORE_CORE_UTILS_SYSTEM_SHA256_H_

#include <string>
#include "mindapi/base/macros.h"

namespace mindspore {
namespace system {
namespace sha256 {
MS_CORE_API std::string GetHashFromString(const std::string &data);
MS_CORE_API std::string GetHashFromFile(const std::string &path);
}  // namespace sha256
}  // namespace system
}  // namespace mindspore
#endif  // MINDSPORE_CORE_UTILS_SYSTEM_SHA256_H_
