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

#ifndef MINDSPORE_CORE_MINDRT_RUNTIME_HQUEUE_H_
#define MINDSPORE_CORE_MINDRT_RUNTIME_HQUEUE_H_
#include <atomic>
#include <vector>

namespace mindspore {
// implement a lock-free queue
// refer to https://www.cs.rochester.edu/u/scott/papers/1996_PODC_queues.pdf

template <typename T>
struct HQNode {
  HQNode() {}
  HQNode(const T &t_, HQNode<T> *n) : t(t_), next(n) {}
  T t;
  std::atomic<HQNode<T> *> next = nullptr;
};

template <typename T>
class HQueue {
 public:
  HQueue(const HQueue &) = delete;
  HQueue &operator=(const HQueue &) = delete;
  HQueue() {}
  virtual ~HQueue() {
    // delete dummy head
    HQNode<T> *node = this->qhead;
    delete node;
  }

  bool Init() {
    HQNode<T> *dummyHead = new HQNode<T>();
    if (!dummyHead) {
      return false;
    }
    qhead = dummyHead;
    qtail = dummyHead;
    return true;
  }

  bool Enqueue(const T &data) {
    HQNode<T> *node = new HQNode<T>(data, nullptr);
    if (!node) {
      return false;
    }

    HQNode<T> *tail = nullptr;
    HQNode<T> *next = nullptr;
    while (true) {
      tail = this->qtail;
      next = tail->next;

      if (tail != this->qtail) {
        continue;
      }

      if (next == nullptr) {
        if (tail->next.compare_exchange_strong(next, node)) {
          break;
        }
      } else {
        this->qtail.compare_exchange_strong(tail, next);
      }
    }
    this->qtail.compare_exchange_weak(tail, node);
    return true;
  }

  bool Dequeue(T *data) {
    HQNode<T> *head = nullptr;
    HQNode<T> *tail = nullptr;
    HQNode<T> *next = nullptr;
    while (true) {
      head = this->qhead;
      tail = this->qtail;
      next = head->next;
      if (head != this->qhead) {
        continue;
      }

      if (head == tail) {
        if (next == nullptr) {
          return false;
        }
        this->qtail.compare_exchange_strong(tail, next);
      } else {
        *data = next->t;
        if (this->qhead.compare_exchange_strong(head, next)) {
          break;
        }
      }
    }

    delete head;
    return true;
  }

  bool Empty() {
    HQNode<T> *head = this->qhead;
    HQNode<T> *tail = this->qtail;
    HQNode<T> *next = head->next;

    if (head == this->qhead && head == tail && next == nullptr) {
      return false;
    }

    return true;
  }

 private:
  std::atomic<HQNode<T> *> qhead;
  std::atomic<HQNode<T> *> qtail;
};

}  // namespace mindspore

#endif  // MINDSPORE_CORE_MINDRT_RUNTIME_HQUEUE_H_
