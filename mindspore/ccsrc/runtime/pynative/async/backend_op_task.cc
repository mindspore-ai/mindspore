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

#include "runtime/pynative/async/backend_op_task.h"

namespace mindspore {
namespace pynative {
void BackendOpRunTask::Run() {
  MS_LOG(DEBUG) << "Wait for build";
  auto build_status = future_.get();
  if (!build_status) {
    MS_LOG(WARNING) << "Op build failed, no need to launch.";
    return;
  }
  MS_EXCEPTION_IF_NULL(run_func_);
  run_func_(context_);
}
}  // namespace pynative
}  // namespace mindspore
