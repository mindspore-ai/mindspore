/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/util/service.h"
#include <thread>

namespace mindspore {
namespace dataset {
Status Service::ServiceStart() {
  do {
    UniqueLock lck(&state_lock_);
    // No-op if it is already up or some other thread is
    // in the process of bring it up.
    if (state_ == STATE::kRunning || state_ == STATE::kStartInProg) {
      return Status::OK();
    }
    // If a stop is in progress, we line up after it
    // is done.
    if (state_ == STATE::kStopInProg) {
      std::this_thread::yield();
    } else {
      state_ = STATE::kStartInProg;
      // At this point, we will let go of the lock. This allow others to proceed.
      lck.Unlock();
      // Call the real implementation from the derived class.
      Status rc = DoServiceStart();
      // If we hit any error, change the state back into the initial state.
      // It is possible that the user may want to drive a clean up by calling
      // ServiceStop but if it will end up in a loop because of the state is still
      // kStartInProg.
      if (rc.IsError()) {
        lck.Lock();
        state_ = STATE::kStopped;
        lck.Unlock();
        return rc;
      }
      // Lock again to change state.
      lck.Lock();
      state_ = STATE::kRunning;
      return Status::OK();
    }
  } while (true);
}

Status Service::ServiceStop() noexcept {
  do {
    UniqueLock lck(&state_lock_);
    // No-op if it is already stopped or some other thread is
    // in the process of shutting it down
    if (state_ == STATE::kStopped || state_ == STATE::kStopInProg) {
      return Status::OK();
    }
    // If a start is in progress, we line up after it
    // is done.
    if (state_ == STATE::kStartInProg) {
      std::this_thread::yield();
    } else {
      state_ = STATE::kStopInProg;
      // At this point, we will let go of the lock. This allows others to proceed.
      lck.Unlock();
      RETURN_IF_NOT_OK(DoServiceStop());
      // Lock again to change state.
      lck.Lock();
      state_ = STATE::kStopped;
      return Status::OK();
    }
  } while (true);
}
}  // namespace dataset
}  // namespace mindspore
