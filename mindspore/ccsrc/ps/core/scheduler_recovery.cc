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

#include "ps/core/scheduler_recovery.h"

namespace mindspore {
namespace ps {
namespace core {
void SchedulerRecovery::Persist(const std::string &key, const std::string &value) {
  MS_EXCEPTION_IF_NULL(recovery_storage_);
  recovery_storage_->Put(key, value);
}

std::string SchedulerRecovery::GetMetadata(const std::string &key) {
  MS_EXCEPTION_IF_NULL(recovery_storage_);
  return recovery_storage_->Get(key, "");
}

bool SchedulerRecovery::Recover() { return true; }
}  // namespace core
}  // namespace ps
}  // namespace mindspore
