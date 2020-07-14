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
#include "minddata/dataset/util/wait_post.h"
#include "minddata/dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
WaitPost::WaitPost() : value_(0) {}

Status WaitPost::Wait() {
  std::unique_lock<std::mutex> lck(mutex_);
  return (wait_cond_.Wait(&lck, [this]() { return value_ != 0; }));
}

void WaitPost::Set() {
  std::unique_lock<std::mutex> lck(mutex_);
  value_ = 1;
  wait_cond_.NotifyAll();
}

void WaitPost::Clear() {
  std::unique_lock<std::mutex> lck(mutex_);
  value_ = 0;
}

Status WaitPost::Register(TaskGroup *vg) { return wait_cond_.Register(vg->GetIntrpService()); }

void WaitPost::ResetIntrpState() { wait_cond_.ResetIntrpState(); }

Status WaitPost::Deregister() { return wait_cond_.Deregister(); }
}  // namespace dataset
}  // namespace mindspore
