/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_QUEUE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_QUEUE_H_

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "./securec.h"
#include "utils/ms_utils.h"
#include "minddata/dataset/util/allocator.h"
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/services.h"
#include "minddata/dataset/util/cond_var.h"
#include "minddata/dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
// A simple thread safe queue using a fixed size array
template <typename T>
class Queue {
 public:
  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;

  explicit Queue(int sz)
      : sz_(sz), arr_(Services::GetAllocator<T>()), head_(0), tail_(0), my_name_(Services::GetUniqueID()) {
    Status rc = arr_.allocate(sz);
    if (rc.IsError()) {
      MS_LOG(ERROR) << "Fail to create a queue.";
      std::terminate();
    } else {
      MS_LOG(DEBUG) << "Create Q with uuid " << my_name_ << " of size " << sz_ << ".";
    }
  }

  virtual ~Queue() { ResetQue(); }

  size_t size() const {
    std::unique_lock<std::mutex> _lock(mux_);
    size_t v = 0;
    if (tail_ >= head_) {
      v = tail_ - head_;
    }
    return v;
  }

  size_t capacity() const {
    std::unique_lock<std::mutex> _lock(mux_);
    return sz_;
  }

  bool empty() const {
    std::unique_lock<std::mutex> _lock(mux_);
    return head_ == tail_;
  }

  void Reset() {
    std::unique_lock<std::mutex> _lock(mux_);
    ResetQue();
    extra_arr_.clear();
  }

  // Producer
  Status Add(const_reference ele) noexcept {
    std::unique_lock<std::mutex> _lock(mux_);
    // Block when full
    Status rc =
      full_cv_.Wait(&_lock, [this]() -> bool { return (SizeWhileHoldingLock() != CapacityWhileHoldingLock()); });
    if (rc.IsOk()) {
      RETURN_IF_NOT_OK(this->AddWhileHoldingLock(ele));
      empty_cv_.NotifyAll();
      _lock.unlock();
    } else {
      empty_cv_.Interrupt();
    }
    return rc;
  }

  Status Add(T &&ele) noexcept {
    std::unique_lock<std::mutex> _lock(mux_);
    // Block when full
    Status rc =
      full_cv_.Wait(&_lock, [this]() -> bool { return (SizeWhileHoldingLock() != CapacityWhileHoldingLock()); });
    if (rc.IsOk()) {
      RETURN_IF_NOT_OK(this->AddWhileHoldingLock(std::forward<T>(ele)));
      empty_cv_.NotifyAll();
      _lock.unlock();
    } else {
      empty_cv_.Interrupt();
    }
    return rc;
  }

  template <typename... Ts>
  Status EmplaceBack(Ts &&... args) noexcept {
    std::unique_lock<std::mutex> _lock(mux_);
    // Block when full
    Status rc =
      full_cv_.Wait(&_lock, [this]() -> bool { return (SizeWhileHoldingLock() != CapacityWhileHoldingLock()); });
    if (rc.IsOk()) {
      auto k = tail_++ % sz_;
      new (arr_[k]) T(std::forward<Ts>(args)...);
      empty_cv_.NotifyAll();
      _lock.unlock();
    } else {
      empty_cv_.Interrupt();
    }
    return rc;
  }

  // Consumer
  virtual Status PopFront(pointer p) {
    std::unique_lock<std::mutex> _lock(mux_);
    // Block when empty
    Status rc = empty_cv_.Wait(&_lock, [this]() -> bool { return !EmptyWhileHoldingLock(); });
    if (rc.IsOk()) {
      RETURN_IF_NOT_OK(this->PopFrontWhileHoldingLock(p, true));
      full_cv_.NotifyAll();
      _lock.unlock();
    } else {
      full_cv_.Interrupt();
    }
    return rc;
  }

  Status Register(TaskGroup *vg) {
    Status rc1 = empty_cv_.Register(vg->GetIntrpService());
    Status rc2 = full_cv_.Register(vg->GetIntrpService());
    if (rc1.IsOk()) {
      return rc2;
    } else {
      return rc1;
    }
  }

  Status Resize(int32_t new_capacity) {
    std::unique_lock<std::mutex> _lock(mux_);
    CHECK_FAIL_RETURN_UNEXPECTED(new_capacity > 0,
                                 "New capacity: " + std::to_string(new_capacity) + ", should be larger than 0");
    RETURN_OK_IF_TRUE(new_capacity == static_cast<int32_t>(CapacityWhileHoldingLock()));
    std::vector<T> queue;
    // pop from the original queue until the new_capacity is full
    for (int32_t i = 0; i < new_capacity; ++i) {
      if (head_ < tail_) {
        // if there are elements left in queue, pop out
        T temp;
        RETURN_IF_NOT_OK(this->PopFrontWhileHoldingLock(&temp, true));
        queue.push_back(temp);
      } else {
        // if there is nothing left in queue, check extra_arr_
        if (!extra_arr_.empty()) {
          // if extra_arr_ is not empty, push to fill the new_capacity
          queue.push_back(extra_arr_[0]);
          extra_arr_.erase(extra_arr_.begin());
        } else {
          // if everything in the queue and extra_arr_ is popped out, break the loop
          break;
        }
      }
    }
    // if there are extra elements in queue, put them to extra_arr_
    while (head_ < tail_) {
      T temp;
      RETURN_IF_NOT_OK(this->PopFrontWhileHoldingLock(&temp, false));
      extra_arr_.push_back(temp);
    }
    this->ResetQue();
    RETURN_IF_NOT_OK(arr_.allocate(new_capacity));
    sz_ = new_capacity;
    for (int32_t i = 0; i < static_cast<int32_t>(queue.size()); ++i) {
      RETURN_IF_NOT_OK(this->AddWhileHoldingLock(queue[i]));
    }
    queue.clear();
    _lock.unlock();
    return Status::OK();
  }

 private:
  size_t sz_;
  MemGuard<T, Allocator<T>> arr_;
  std::vector<T> extra_arr_;  // used to store extra elements after reducing capacity, will not be changed by Add,
                              // will pop when there is a space in queue (by PopFront or Resize)
  size_t head_;
  size_t tail_;
  std::string my_name_;
  mutable std::mutex mux_;
  CondVar empty_cv_;
  CondVar full_cv_;

  // Helper function for Add, must be called when holding a lock
  Status AddWhileHoldingLock(const_reference ele) {
    auto k = tail_++ % sz_;
    *(arr_[k]) = ele;
    return Status::OK();
  }

  // Helper function for Add, must be called when holding a lock
  Status AddWhileHoldingLock(T &&ele) {
    auto k = tail_++ % sz_;
    *(arr_[k]) = std::forward<T>(ele);
    return Status::OK();
  }

  // Helper function for PopFront, must be called when holding a lock
  Status PopFrontWhileHoldingLock(pointer p, bool clean_extra) {
    auto k = head_++ % sz_;
    *p = std::move(*(arr_[k]));
    if (!extra_arr_.empty() && clean_extra) {
      RETURN_IF_NOT_OK(this->AddWhileHoldingLock(std::forward<T>(extra_arr_[0])));
      extra_arr_.erase(extra_arr_.begin());
    }
    return Status::OK();
  }

  void ResetQue() noexcept {
    while (head_ < tail_) {
      T val;
      this->PopFrontWhileHoldingLock(&val, false);
      MS_LOG(DEBUG) << "Address of val: " << &val;
    }
    empty_cv_.ResetIntrpState();
    full_cv_.ResetIntrpState();
    head_ = 0;
    tail_ = 0;
  }

  size_t SizeWhileHoldingLock() const {
    size_t v = 0;
    if (tail_ >= head_) {
      v = tail_ - head_;
    }
    return v;
  }

  size_t CapacityWhileHoldingLock() const { return sz_; }

  bool EmptyWhileHoldingLock() const { return head_ == tail_; }
};

// A container of queues with [] operator accessors.  Basically this is a wrapper over of a vector of queues
// to help abstract/simplify code that is maintaining multiple queues.
template <typename T>
class QueueList {
 public:
  QueueList() {}

  void Init(int num_queues, int capacity) {
    (void)queue_list_.reserve(num_queues);
    for (int i = 0; i < num_queues; i++) {
      (void)queue_list_.emplace_back(std::make_unique<Queue<T>>(capacity));
    }
  }

  Status Register(TaskGroup *vg) {
    if (vg == nullptr) {
      RETURN_STATUS_UNEXPECTED("Null task group during QueueList registration.");
    }
    for (int i = 0; i < queue_list_.size(); ++i) {
      RETURN_IF_NOT_OK(queue_list_[i]->Register(vg));
    }
    return Status::OK();
  }

  auto size() const {
    std::unique_lock<std::mutex> _lock(mux_);
    return queue_list_.size();
  }

  std::unique_ptr<Queue<T>> &operator[](const int index) {
    std::unique_lock<std::mutex> _lock(mux_);
    return queue_list_[index];
  }

  const std::unique_ptr<Queue<T>> &operator[](const int index) const {
    std::unique_lock<std::mutex> _lock(mux_);
    return queue_list_[index];
  }

  ~QueueList() = default;

  Status AddQueue(TaskGroup *vg) {
    std::unique_lock<std::mutex> _lock(mux_);
    (void)queue_list_.emplace_back(std::make_unique<Queue<T>>(queue_list_[0]->capacity()));
    return queue_list_[queue_list_.size() - 1]->Register(vg);
  }
  Status RemoveLastQueue() {
    std::unique_lock<std::mutex> _lock(mux_);
    CHECK_FAIL_RETURN_UNEXPECTED(queue_list_.size() > 1, "Cannot remove more than the current queues.");
    (void)queue_list_.pop_back();
    return Status::OK();
  }

 private:
  // Queue contains non-copyable objects, so it cannot be added to a vector due to the vector
  // requirement that objects must have copy semantics.  To resolve this, we use a vector of unique
  // pointers.  This allows us to provide dynamic creation of queues in a container.
  std::vector<std::unique_ptr<Queue<T>>> queue_list_;

  mutable std::mutex mux_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_QUEUE_H_
