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
#include "dataset/util/lock.h"

namespace mindspore {
namespace dataset {
void SpinLock::Lock() {
  while (true) {
    int expected = kUnlocked;
    if (val_.compare_exchange_weak(expected, kLocked)) {
      break;
    }
  }
}

bool SpinLock::TryLock() {
  int expected = kUnlocked;
  return val_.compare_exchange_strong(expected, kLocked);
}

void SpinLock::Unlock() noexcept { val_.store(kUnlocked); }

void RWLock::LockShared() {
  std::unique_lock<std::mutex> lck(mtx_);
  waiting_readers_ += 1;
  read_cv_.wait(lck, [this]() { return (waiting_writers_ == 0 && status_ >= 0); });
  waiting_readers_ -= 1;
  status_ += 1;
}

void RWLock::Unlock() noexcept {
  std::unique_lock<std::mutex> lck(mtx_);
  if (status_ == -1) {
    // I am the writer. By definition, no other writer nor reader.
    status_ = 0;
  } else if (status_ > 0) {
    // One less reader
    status_ -= 1;
  }
  // Wake up writer only if there is no reader.
  if (waiting_writers_ > 0) {
    if (status_ == 0) {
      write_cv_.notify_one();
    }
  } else {
    read_cv_.notify_all();
  }
}

void RWLock::Upgrade() {
  std::unique_lock<std::mutex> lck(mtx_);
  DS_ASSERT(status_);
  if (status_ == -1) {
    // I am a writer already.
    return;
  } else if (status_ == 1) {
    // If I am the only reader. Just change the status.
    status_ = -1;
    return;
  } else {
    // In all other cases, let of the shared lock and relock in exclusive.
    lck.unlock();
    this->Unlock();
    this->LockExclusive();
  }
}

void RWLock::Downgrade() {
  std::unique_lock<std::mutex> lck(mtx_);
  DS_ASSERT(status_);
  if (status_ == -1) {
    // If there are no other writers waiting, just change the status
    if (waiting_writers_ == 0) {
      status_ = 1;
    } else {
      // Otherwise just unlock and relock in shared
      lck.unlock();
      this->Unlock();
      this->LockShared();
    }
  } else if (status_ > 0) {
    return;
  }
}

SharedLock::SharedLock(RWLock *rw) : rw_(rw), ownlock_(false) {
  rw_->LockShared();
  ownlock_ = true;
}

SharedLock::~SharedLock() {
  if (ownlock_) {
    rw_->Unlock();
    ownlock_ = false;
  }
  rw_ = nullptr;
}

void SharedLock::Unlock() {
  DS_ASSERT(ownlock_ == true);
  rw_->Unlock();
  ownlock_ = false;
}

void SharedLock::Lock() {
  DS_ASSERT(ownlock_ == false);
  rw_->LockShared();
  ownlock_ = true;
}

void SharedLock::Upgrade() {
  DS_ASSERT(ownlock_ == true);
  rw_->Upgrade();
}

void SharedLock::Downgrade() {
  DS_ASSERT(ownlock_ == true);
  rw_->Downgrade();
}

UniqueLock::UniqueLock(RWLock *rw) : rw_(rw), ownlock_(false) {
  rw_->LockExclusive();
  ownlock_ = true;
}

UniqueLock::~UniqueLock() {
  if (ownlock_) {
    rw_->Unlock();
    ownlock_ = false;
  }
  rw_ = nullptr;
}

void UniqueLock::Unlock() {
  DS_ASSERT(ownlock_ == true);
  rw_->Unlock();
  ownlock_ = false;
}

void UniqueLock::Lock() {
  DS_ASSERT(ownlock_ == false);
  rw_->LockExclusive();
  ownlock_ = true;
}

LockGuard::LockGuard(SpinLock *lock) : lck_(lock), own_lock_(false) {
  lck_->Lock();
  own_lock_ = true;
}

LockGuard::~LockGuard() {
  if (own_lock_) {
    lck_->Unlock();
    own_lock_ = false;
  }
  lck_ = nullptr;
}

void LockGuard::Unlock() {
  DS_ASSERT(own_lock_);
  lck_->Unlock();
  own_lock_ = false;
}

void LockGuard::Lock() {
  DS_ASSERT(own_lock_ == false);
  lck_->Lock();
  own_lock_ = true;
}
}  // namespace dataset
}  // namespace mindspore
