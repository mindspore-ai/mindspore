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
    size_t v = tail_ - head_;
    return (v >= 0) ? v : 0;
  }

  size_t capacity() const { return sz_; }

  bool empty() const { return head_ == tail_; }

  void Reset() { ResetQue(); }

  // Producer
  Status Add(const_reference ele) noexcept {
    std::unique_lock<std::mutex> _lock(mux_);
    // Block when full
    Status rc = full_cv_.Wait(&_lock, [this]() -> bool { return (size() != capacity()); });
    if (rc.IsOk()) {
      auto k = tail_++ % sz_;
      *(arr_[k]) = ele;
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
    Status rc = full_cv_.Wait(&_lock, [this]() -> bool { return (size() != capacity()); });
    if (rc.IsOk()) {
      auto k = tail_++ % sz_;
      *(arr_[k]) = std::forward<T>(ele);
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
    Status rc = full_cv_.Wait(&_lock, [this]() -> bool { return (size() != capacity()); });
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
  Status PopFront(pointer p) {
    std::unique_lock<std::mutex> _lock(mux_);
    // Block when empty
    Status rc = empty_cv_.Wait(&_lock, [this]() -> bool { return !empty(); });
    if (rc.IsOk()) {
      auto k = head_++ % sz_;
      *p = std::move(*(arr_[k]));
      full_cv_.NotifyAll();
      _lock.unlock();
    } else {
      full_cv_.Interrupt();
    }
    return rc;
  }

  void ResetQue() noexcept {
    std::unique_lock<std::mutex> _lock(mux_);
    // If there are elements in the queue, drain them. We won't call PopFront directly
    // because we have got the lock already. We will deadlock if we call PopFront
    for (auto i = head_; i < tail_; ++i) {
      auto k = i % sz_;
      auto val = std::move(*(arr_[k]));
      // Let val go out of scope and its destructor will be invoked automatically.
      // But our compiler may complain val is not in use. So let's do some useless
      // stuff.
      MS_LOG(DEBUG) << "Address of val: " << &val;
    }
    empty_cv_.ResetIntrpState();
    full_cv_.ResetIntrpState();
    head_ = 0;
    tail_ = 0;
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

 private:
  size_t sz_;
  MemGuard<T, Allocator<T>> arr_;
  size_t head_;
  size_t tail_;
  std::string my_name_;
  std::mutex mux_;
  CondVar empty_cv_;
  CondVar full_cv_;
};

// A container of queues with [] operator accessors.  Basically this is a wrapper over of a vector of queues
// to help abstract/simplify code that is maintaining multiple queues.
template <typename T>
class QueueList {
 public:
  QueueList() {}

  void Init(int num_queues, int capacity) {
    queue_list_.reserve(num_queues);
    for (int i = 0; i < num_queues; i++) {
      queue_list_.emplace_back(std::make_unique<Queue<T>>(capacity));
    }
  }

  Status Register(TaskGroup *vg) {
    if (vg == nullptr) {
      return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                    "Null task group during QueueList registration.");
    }
    for (int i = 0; i < queue_list_.size(); ++i) {
      RETURN_IF_NOT_OK(queue_list_[i]->Register(vg));
    }
    return Status::OK();
  }

  auto size() const { return queue_list_.size(); }

  std::unique_ptr<Queue<T>> &operator[](const int index) { return queue_list_[index]; }

  const std::unique_ptr<Queue<T>> &operator[](const int index) const { return queue_list_[index]; }

  ~QueueList() = default;

 private:
  // Queue contains non-copyable objects, so it cannot be added to a vector due to the vector
  // requirement that objects must have copy semantics.  To resolve this, we use a vector of unique
  // pointers.  This allows us to provide dynamic creation of queues in a container.
  std::vector<std::unique_ptr<Queue<T>>> queue_list_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_QUEUE_H_
