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

#include "runtime/framework/actor/actor_common.h"
#include <unistd.h>
#ifdef __WIN32__
#include <windows.h>
#endif

namespace mindspore {
namespace runtime {
int64_t GetMaxThreadNum() {
#ifdef __WIN32__
  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);
  auto max_thread_num = sys_info.dwNumberOfProcessors;
#else
  auto max_thread_num = sysconf(_SC_NPROCESSORS_ONLN);
#endif

  return max_thread_num;
}

}  // namespace runtime
}  // namespace mindspore
