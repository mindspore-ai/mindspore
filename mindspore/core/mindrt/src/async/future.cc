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

#include "async/future.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace internal {

void Waitf(const AID &aid) {
  mindspore::Terminate(aid);
  MS_LOG(WARNING) << "WaitFor is timeout.";
}

void Wait(const AID &aid, const mindspore::Timer &timer) {
  mindspore::TimerTools::Cancel(timer);
  mindspore::Terminate(aid);
}

}  // namespace internal
}  // namespace mindspore
