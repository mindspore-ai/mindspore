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
#include "dataset/util/cond_var.h"
#include <utility>
#include "dataset/util/services.h"
#include "dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
CondVar::CondVar() : svc_(nullptr), my_name_(std::move(Services::GetUniqueID())) {}

Status CondVar::Wait(std::unique_lock<std::mutex> *lck, const std::function<bool()> &pred) {
  // Append an additional condition on top of the given predicate.
  // We will also bail out if this cv got interrupted.
  auto f = [this, &pred]() -> bool { return (pred() || (CurState() == State::kInterrupted)); };
  // If we have interrupt service, just wait on the cv unconditionally.
  // Otherwise fall back to the old way of checking interrupt.
  if (svc_) {
    cv_.wait(*lck, f);
    if (CurState() == State::kInterrupted) {
      Task *my_task = TaskManager::FindMe();
      if (my_task->IsMasterThread() && my_task->CaughtSevereException()) {
        return TaskManager::GetMasterThreadRc();
      } else {
        return Status(StatusCode::kInterrupted);
      }
    }
  } else {
    RETURN_IF_NOT_OK(interruptible_wait(&cv_, lck, pred));
    if (CurState() == State::kInterrupted) {
      return Status(StatusCode::kInterrupted);
    }
  }
  return Status::OK();
}

CondVar::~CondVar() noexcept {
  if (svc_ != nullptr) {
    (void)svc_->Deregister(my_name_);
    svc_ = nullptr;
  }
}

void CondVar::NotifyOne() noexcept { cv_.notify_one(); }

void CondVar::NotifyAll() noexcept { cv_.notify_all(); }

Status CondVar::Register(std::shared_ptr<IntrpService> svc) {
  Status rc = svc->Register(my_name_, this);
  if (rc.IsOk()) {
    svc_ = svc;
  }
  return rc;
}

Status CondVar::Interrupt() {
  RETURN_IF_NOT_OK(IntrpResource::Interrupt());
  cv_.notify_all();
  return Status::OK();
}

std::string CondVar::my_name() const { return my_name_; }

Status CondVar::Deregister() {
  if (svc_) {
    Status rc = svc_->Deregister(my_name_);
    svc_ = nullptr;
    return rc;
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
