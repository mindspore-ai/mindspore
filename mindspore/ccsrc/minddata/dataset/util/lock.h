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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_LOCK_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_LOCK_H_

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace mindspore {
namespace dataset {
class SpinLock {
 public:
  void Lock();

  bool TryLock();

  void Unlock() noexcept;

  SpinLock() : val_(kUnlocked) {}

  SpinLock(const SpinLock &) = delete;

  SpinLock(SpinLock &&) = delete;

  ~SpinLock() = default;

  SpinLock &operator=(const SpinLock &) = delete;

  SpinLock &operator=(SpinLock &&) = delete;

 private:
  static constexpr int kUnlocked = 0;
  static constexpr int kLocked = 1;
  std::atomic<int> val_;
};

// C++11 has no shared mutex. The following class is an alternative. It favors writer and is suitable for the case
// where writer is rare.
class RWLock {
 public:
  RWLock() : status_(0), waiting_readers_(0), waiting_writers_(0) {}

  RWLock(const RWLock &) = delete;

  RWLock(RWLock &&) = delete;

  ~RWLock() = default;

  RWLock &operator=(const RWLock &) = delete;

  RWLock &operator=(RWLock &&) = delete;

  void LockShared();

  void LockExclusive() {
    std::unique_lock<std::mutex> lck(mtx_);
    waiting_writers_ += 1;
    write_cv_.wait(lck, [this]() { return status_ == 0; });
    waiting_writers_ -= 1;
    status_ = -1;
  }

  void Unlock() noexcept;

  // Upgrade a shared lock to exclusive lock
  void Upgrade();

  // Downgrade an exclusive lock to shared lock
  void Downgrade();

 private:
  // -1    : one writer
  // 0     : no reader and no writer
  // n > 0 : n reader
  int32_t status_;
  int32_t waiting_readers_;
  int32_t waiting_writers_;
  std::mutex mtx_;
  std::condition_variable read_cv_;
  std::condition_variable write_cv_;
};

// A Wrapper for RWLock. The destructor will release the lock if we own it.
class SharedLock {
 public:
  explicit SharedLock(RWLock *rw);

  ~SharedLock();

  SharedLock(const SharedLock &) = delete;

  SharedLock(SharedLock &&) = delete;

  SharedLock &operator=(const SharedLock &) = delete;

  SharedLock &operator=(SharedLock &&) = delete;

  void Unlock();

  void Lock();

  void Upgrade();

  void Downgrade();

 private:
  RWLock *rw_;
  bool ownlock_;
};

class UniqueLock {
 public:
  explicit UniqueLock(RWLock *rw);

  ~UniqueLock();

  UniqueLock(const UniqueLock &) = delete;

  UniqueLock(UniqueLock &&) = delete;

  UniqueLock &operator=(const UniqueLock &) = delete;

  UniqueLock &operator=(UniqueLock &&) = delete;

  void Unlock();

  void Lock();

 private:
  RWLock *rw_;
  bool ownlock_;
};

class LockGuard {
 public:
  explicit LockGuard(SpinLock *lock);

  ~LockGuard();

  LockGuard(const LockGuard &) = delete;

  LockGuard(LockGuard &&) = delete;

  LockGuard &operator=(const LockGuard &) = delete;

  LockGuard &operator=(LockGuard &&) = delete;

  void Unlock();

  void Lock();

 private:
  SpinLock *lck_;
  bool own_lock_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_LOCK_H_
