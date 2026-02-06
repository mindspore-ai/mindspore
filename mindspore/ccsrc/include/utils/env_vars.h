/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_ENV_VARS_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_ENV_VARS_H_

#include "include/utils/visible.h"

namespace mindspore {
namespace common {
// get slice_size for print, tensordump, tensorsummary, etc.
COMMON_EXPORT int GetDumpSliceSize();

// get wait_time for print, tensordump, tensorsummary, etc.
COMMON_EXPORT int GetDumpWaitTime();
}  // namespace common
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_ENV_VARS_H_
