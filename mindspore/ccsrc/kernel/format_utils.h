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
#ifndef MINDSPORE_CCSRC_KERNEL_FORMAT_UTILS_H_
#define MINDSPORE_CCSRC_KERNEL_FORMAT_UTILS_H_

#include <string>
#include "include/backend/visible.h"
#include "include/api/format.h"

namespace mindspore {
namespace kernel {
BACKEND_EXPORT Format GetFormatFromStrToEnum(const std::string &format_str);
BACKEND_EXPORT std::string GetFormatFromEnumToStr(Format format);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_FORMAT_UTILS_H_
