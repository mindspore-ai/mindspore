/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_QUEUE_MAP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_QUEUE_MAP_H_

#include <atomic>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include "minddata/dataset/util/allocator.h"
#include "minddata/dataset/util/system_pool.h"
#include "minddata/dataset/util/semaphore.h"
#include "minddata/dataset/util/services.h"
namespace mindspore {
namespace dataset {
template <typename K, typename T>
/// \brief QueueMap is like a Queue but instead of there is a map of deque<T>.
/// Consumer will block if the corresponding deque is empty.
/// Producer can add an element of type T with key of type K to the map and
/// wake up any waiting consumer.
/// \tparam K key type
/// \tparam T payload of the map
class QueueMap {
 public:
  using key_type = K;
  using value_type = T;

  QueueMap() : num_rows_(0) {}
  virtual ~QueueMap() = default;

  /// Add an element <key, T> to the map and wake up any consumer that is waiting
  /// \param key
  /// \param payload
  /// \return Status object
  virtual Status Add(key_type key, T &&payload) {
    RequestQueue *rq = nullptr;
    RETURN_IF_NOT_OK(GetRq(key, &rq));
    RETURN_IF_NOT_OK(rq->WakeUpAny(std::move(payload)));
    ++num_rows_;
    return Status::OK();
  }

  /// Pop the front of the deque with key. Block if the deque is empty.
  virtual Status PopFront(key_type key, T *out) {
    RequestQueue *rq = nullptr;
    RETURN_IF_NOT_OK(GetRq(key, &rq));
    RETURN_IF_NOT_OK(rq->Wait(out));
    --num_rows_;
    return Status::OK();
  }

  /// Get the number of elements in the container
  /// \return The number of elements in the container
  int64_t size() const { return num_rows_; }

  /// \return if the container is empty
  bool empty() const { return num_rows_ == 0; }

  /// Print out some useful information about the container
  friend std::ostream &operator<<(std::ostream &out, const QueueMap &qm) {
    std::unique_lock<std::mutex> lck(qm.mux_);
    out << "Number of elements: " << qm.num_rows_ << "\n";
    out << "Dumping internal info:\n";
    int64_t k = 0;
    constexpr int64_t line_breaks_number = 6;
    for (auto &it : qm.all_) {
      auto key = it.first;
      const RequestQueue *rq = it.second.GetPointer();
      out << "(k:" << key << "," << *rq << ") ";
      ++k;
      if (k % line_breaks_number == 0) {
        out << "\n";
      }
    }
    return out;
  }

 protected:
  /// This is a handshake structure between producer and consumer
  class RequestQueue {
   public:
    RequestQueue() : use_count_(0) {}
    ~RequestQueue() = default;

    Status Wait(T *out) {
      RETURN_UNEXPECTED_IF_NULL(out);
      // Block until the missing row is in the pool.
      RETURN_IF_NOT_OK(use_count_.P());
      std::unique_lock<std::mutex> lck(dq_mux_);
      CHECK_FAIL_RETURN_UNEXPECTED(!row_.empty(), "Programming error");
      *out = std::move(row_.front());
      row_.pop_front();
      return Status::OK();
    }

    Status WakeUpAny(T &&row) {
      std::unique_lock<std::mutex> lck(dq_mux_);
      row_.push_back(std::move(row));
      // Bump up the use count by 1. This wake up any parallel worker which is waiting
      // for this row.
      use_count_.V();
      return Status::OK();
    }

    friend std::ostream &operator<<(std::ostream &out, const RequestQueue &rq) {
      out << "sz:" << rq.row_.size() << ",uc:" << rq.use_count_.Peek();
      return out;
    }

   private:
    mutable std::mutex dq_mux_;
    Semaphore use_count_;
    std::deque<T> row_;
  };

  /// Create or locate an element with matching key
  /// \param key
  /// \param out
  /// \return Status object
  Status GetRq(key_type key, RequestQueue **out) {
    RETURN_UNEXPECTED_IF_NULL(out);
    std::unique_lock<std::mutex> lck(mux_);
    auto it = all_.find(key);
    if (it != all_.end()) {
      *out = it->second.GetMutablePointer();
    } else {
      // We will create a new one.
      auto alloc = SystemPool::GetAllocator<RequestQueue>();
      auto r = all_.emplace(key, MemGuard<RequestQueue, Allocator<RequestQueue>>(alloc));
      if (r.second) {
        auto &mem = r.first->second;
        RETURN_IF_NOT_OK(mem.allocate(1));
        *out = mem.GetMutablePointer();
      } else {
        RETURN_STATUS_UNEXPECTED("Map insert fail.");
      }
    }
    return Status::OK();
  }

 private:
  mutable std::mutex mux_;
  std::map<K, MemGuard<RequestQueue, Allocator<RequestQueue>>> all_;
  std::atomic<int64_t> num_rows_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_QUEUE_MAP_H_
