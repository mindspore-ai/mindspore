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
#include <exception>
#include <utility>
#include "dataset/util/services.h"
#include "dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
CondVar::CondVar() : svc_(nullptr), my_name_(Services::GetUniqueID()) {}

Status CondVar::Wait(std::unique_lock<std::mutex> *lck, const std::function<bool()> &pred) {
  try {
    if (svc_ != nullptr) {
      // If this cv registers with a global resource tracking, then wait unconditionally.
      auto f = [this, &pred]() -> bool { return (pred() || this->Interrupted()); };
      cv_.wait(*lck, f);
      // If we are interrupted, override the return value if this is the master thread.
      // Master thread is being interrupted mostly because of some thread is reporting error.
      RETURN_IF_NOT_OK(Task::OverrideInterruptRc(this->GetInterruptStatus()));
    } else {
      // Otherwise we wake up once a while to check for interrupt (for this thread).
      auto f = [&pred]() -> bool { return (pred() || this_thread::is_interrupted()); };
      while (!f()) {
        (void)cv_.wait_for(*lck, std::chrono::milliseconds(1));
      }
      RETURN_IF_INTERRUPTED();
    }
  } catch (const std::exception &e) {
    RETURN_STATUS_UNEXPECTED(e.what());
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

void CondVar::Interrupt() {
  IntrpResource::Interrupt();
  cv_.notify_all();
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
