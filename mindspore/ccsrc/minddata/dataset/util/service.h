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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_SERVICE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_SERVICE_H_

#include <atomic>
#include "minddata/dataset/util/lock.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class Service {
 public:
  enum class STATE : int { kStartInProg = 1, kRunning, kStopInProg, kStopped };

  Service() : state_(STATE::kStopped) {}

  Service(const Service &) = delete;

  Service &operator=(const Service &) = delete;

  virtual ~Service() {}

  STATE ServiceState() const { return state_; }

  virtual Status DoServiceStart() = 0;

  virtual Status DoServiceStop() = 0;

  Status ServiceStart();

  Status ServiceStop() noexcept;

 protected:
  STATE state_;
  RWLock state_lock_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_SERVICE_H_
