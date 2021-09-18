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

#include "utils/log_adapter.h"

namespace mindspore {
std::map<void **, std::thread *> acl_handle_map;
// set default log level to WARNING for all sub modules
int g_ms_submodule_log_levels[NUM_SUBMODUES] = {WARNING};
#if defined(_WIN32) || defined(_WIN64)
enum MsLogLevel this_thread_max_log_level = EXCEPTION;
#else
thread_local enum MsLogLevel this_thread_max_log_level = EXCEPTION;
#endif
}  // namespace mindspore
