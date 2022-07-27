/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_UTILS_LABEL_H_
#define MINDSPORE_CORE_UTILS_LABEL_H_
#include <iostream>
#include <memory>
#include <string>
#include "utils/hash_map.h"
#include "ir/anf.h"

namespace mindspore {
namespace label_manage {
enum class TraceLabelType { kShortSymbol, kFullName, kWithUniqueId };
MS_CORE_API TraceLabelType GetGlobalTraceLabelType();
MS_CORE_API std::string Label(const DebugInfoPtr &debug_info,
                              TraceLabelType trace_label = TraceLabelType::kShortSymbol);
}  // namespace label_manage
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_LABEL_H_
