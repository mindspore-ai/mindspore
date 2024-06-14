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

#include "runtime/pipeline/task/task.h"
#include <atomic>

namespace mindspore {
namespace runtime {
uint64_t AsyncTask::MakeId() {
  static std::atomic<uint64_t> last_id{1};
  return last_id.fetch_add(1, std::memory_order_relaxed);
}
}  // namespace runtime
}  // namespace mindspore
