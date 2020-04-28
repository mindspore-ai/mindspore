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
#ifndef DATASET_UTIL_WAIT_POST_H_
#define DATASET_UTIL_WAIT_POST_H_

#include <mutex>
#include "dataset/util/cond_var.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
class TaskGroup;

class WaitPost {
 public:
  WaitPost();

  ~WaitPost() = default;

  Status Wait();

  void Set();

  void Clear();

  Status Register(TaskGroup *vg);

  Status Deregister();

  void ResetIntrpState();

 private:
  std::mutex mutex_;
  CondVar wait_cond_;
  int value_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_UTIL_WAIT_POST_H_
