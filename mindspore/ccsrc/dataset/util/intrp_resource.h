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
#ifndef DATASET_UTIL_INTRP_RESOURCE_H_
#define DATASET_UTIL_INTRP_RESOURCE_H_

#include <atomic>
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
class IntrpResource {
 public:
  enum class State : int { kRunning, kInterrupted };

  IntrpResource() : st_(State::kRunning) {}

  virtual ~IntrpResource() = default;

  virtual Status Interrupt() {
    st_ = State::kInterrupted;
    return Status::OK();
  }

  virtual void ResetIntrpState() { st_ = State::kRunning; }

  State CurState() const { return st_; }

  bool Interrupted() const { return CurState() == State::kInterrupted; }

 protected:
  std::atomic<State> st_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_UTIL_INTRP_RESOURCE_H_
