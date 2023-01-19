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
#include "minddata/dataset/util/intrp_service.h"
#include <sstream>
#include "minddata/dataset/util/services.h"
#include "minddata/dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
const int64_t kServiceRetryGetUniqueIdInterVal = 10;
IntrpService::IntrpService() try : high_water_mark_(0) { (void)ServiceStart(); } catch (const std::exception &e) {
  MS_LOG(ERROR) << "Interrupt service failed: " << e.what() << ".";
  std::terminate();
}

IntrpService::~IntrpService() noexcept {
  MS_LOG(DEBUG) << "Number of registered resources is " << high_water_mark_ << ".";
  if (!all_intrp_resources_.empty()) {
    try {
      InterruptAll();
    } catch (const std::exception &e) {
      // Ignore all error as we can't throw in the destructor.
    }
  }
  (void)ServiceStop();
}

Status IntrpService::Register(std::string *name, IntrpResource *res) {
  SharedLock stateLck(&state_lock_);
  // Now double check the state
  if (ServiceState() != STATE::kRunning) {
    RETURN_STATUS_ERROR(StatusCode::kMDInterrupted, "Interrupt service is shutting down");
  } else {
    std::lock_guard<std::mutex> lck(mutex_);
    try {
      std::ostringstream ss;
      std::string uuid = std::string("");
      ss << this_thread::get_id();
      MS_LOG(DEBUG) << "Register resource with name " << *name << ". Thread ID " << ss.str() << ".";
      auto it = all_intrp_resources_.emplace(*name, res);
      while (it.second == false) {
        uuid = Services::GetUniqueID();
        it = all_intrp_resources_.emplace(uuid, res);
        MS_LOG(INFO) << "The name(" << *name << ") of register resource is duplicate, get new uuid: " << uuid;
        std::this_thread::sleep_for(std::chrono::milliseconds(kServiceRetryGetUniqueIdInterVal));
      }
      if (!uuid.empty()) {
        *name = uuid;
      }
      high_water_mark_++;
    } catch (std::exception &e) {
      RETURN_STATUS_UNEXPECTED(e.what());
    }
  }
  return Status::OK();
}

Status IntrpService::Deregister(const std::string &name) noexcept {
  std::lock_guard<std::mutex> lck(mutex_);
  try {
    std::ostringstream ss;
    ss << this_thread::get_id();
    MS_LOG(DEBUG) << "De-register resource with name " << name << ". Thread ID is " << ss.str() << ".";
    auto n = all_intrp_resources_.erase(name);
    if (n == 0) {
      MS_LOG(INFO) << "Key " << name << " not found.";
    }
  } catch (std::exception &e) {
    RETURN_STATUS_UNEXPECTED(e.what());
  }
  return Status::OK();
}

void IntrpService::InterruptAll() noexcept {
  std::lock_guard<std::mutex> lck(mutex_);
  for (auto const &it : all_intrp_resources_) {
    std::string kName = it.first;
    try {
      it.second->Interrupt();
    } catch (const std::exception &e) {
      // continue the clean up.
    }
  }
}
}  // namespace dataset
}  // namespace mindspore
