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
#ifndef DATASET_UTIL_SEMAPHORE_H_
#define DATASET_UTIL_SEMAPHORE_H_

#include "dataset/util/cond_var.h"

namespace mindspore {
namespace dataset {
class TaskGroup;

class Semaphore {
 public:
  explicit Semaphore(int init) : value_(init) {}

  virtual ~Semaphore() {}

  Status P();

  void V();

  void Register(TaskGroup *vg);

  Status Deregister();

  void ResetIntrpState();

 private:
  int value_;

  std::mutex mutex_;
  CondVar wait_cond_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_UTIL_SEMAPHORE_H_
