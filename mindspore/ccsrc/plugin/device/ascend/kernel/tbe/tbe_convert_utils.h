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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_COMMON_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_COMMON_UTILS_H_

#include <string>
#include "kernel/kernel.h"
#include "base/base.h"
#include "ir/dtype/type.h"

namespace mindspore {
namespace kernel {
namespace tbe {
constexpr auto kProcessorAiCore = "aicore";
BACKEND_EXPORT TypeId DtypeToTypeId(const std::string &dtypes);

std::string TypeIdToString(TypeId type_id);

size_t GetDtypeNbyte(const std::string &dtypes);
}  // namespace tbe
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_COMMON_UTILS_H_
