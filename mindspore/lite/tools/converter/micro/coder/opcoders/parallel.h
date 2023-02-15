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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_PARALLEL_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_PARALLEL_H_

namespace mindspore::lite::micro {
constexpr auto kDefaultTaskId = 0;

constexpr auto kMaxThreadNumSupported = 4;
constexpr auto kDefaultThreadNum = 1;

// ParallelLaunch is defined in thread_pool
constexpr auto kParallelLaunch = "ParallelLaunch";

// g_thread_pool and g_thread_num are global variable,
// and g_thread_num is equal to GetCurrentThreadNum
extern const char *gThreadNum;
void InitThread(int model_index);
void FreeThread();

// args represents the parameters required for operator to run
constexpr auto kRunArgs = "args";
constexpr auto kRunArgsAddr = "(void *)&args";

constexpr float kLhsScale = 0;
constexpr float kRhsScale = 1;
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_PARALLEL_H_
