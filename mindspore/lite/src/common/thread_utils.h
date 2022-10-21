/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_COMMON_THREAD_UTILS_H_
#define MINDSPORE_LITE_SRC_COMMON_THREAD_UTILS_H_

#ifdef __linux__
#include "include/api/status.h"
namespace mindspore {
namespace lite {
constexpr int kProcessSuccess = 0;
constexpr int kProcessFailed = 1;
constexpr int kSingleThread = 1;

Status CheckPidStatus(pid_t pid);
int GetNumThreads();
}  // namespace lite
}  // namespace mindspore
#endif
#endif  // MINDSPORE_LITE_SRC_COMMON_THREAD_UTILS_H_
