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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_SEMAPHORE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_SEMAPHORE_H_

#include "minddata/dataset/util/cond_var.h"

namespace mindspore {
namespace dataset {
class TaskGroup;

/// \brief A counting semaphore. There are two external functions P and V. P decrements the internal count and will be
/// blocked if the count is 0 (zero). V increments the internal count and wake up one of the waiters.
class Semaphore {
 public:
  /// \brief Constructor
  /// \param init Initial value of the internal counter.
  explicit Semaphore(int init) : value_(init) {}

  virtual ~Semaphore() {}
  /// \brief Decrement the internal counter. Will be blocked if the value is 0.
  /// \return Error code. Can get interrupt.
  Status P();
  /// \brief Increment the internal counter. Wake up on of the waiters if any.
  void V();
  /// \brief Peek the internal value
  /// \return The internal value
  int Peek() const;
  Status Register(TaskGroup *vg);
  Status Deregister();
  void ResetIntrpState();

 private:
  int value_;

  std::mutex mutex_;
  CondVar wait_cond_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_SEMAPHORE_H_
