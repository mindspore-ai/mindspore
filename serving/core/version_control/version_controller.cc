/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "core/version_control/version_controller.h"

#include <string>
#include <iostream>
#include <ctime>
#include <memory>
#include "util/file_system_operation.h"
#include "mindspore/ccsrc/utils/log_adapter.h"
#include "core/server.h"

namespace mindspore {
namespace serving {
volatile bool stop_poll = false;

std::string GetVersionFromPath(const std::string &path) {
  std::string new_path = path;
  if (path.back() == '/') {
    new_path = path.substr(0, path.size() - 1);
  }

  std::string::size_type index = new_path.find_last_of("/");
  std::string version = new_path.substr(index + 1);
  return version;
}

void PeriodicFunction::operator()() {
  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(poll_model_wait_seconds_ * 1000));
    std::vector<std::string> SubDirs = GetAllSubDirs(models_path_);

    if (version_control_strategy_ == VersionController::VersionControllerStrategy::kLastest) {
      auto path = SubDirs.empty() ? models_path_ : SubDirs.back();
      std::string model_version = GetVersionFromPath(path);
      time_t last_update_time = GetModifyTime(path);
      if (model_version != valid_models_.back()->GetModelVersion()) {
        MindSporeModelPtr model_ptr = std::make_shared<MindSporeModel>(valid_models_.front()->GetModelName(), path,
                                                                       model_version, last_update_time);
        valid_models_.back() = model_ptr;
        Session::Instance().Warmup(valid_models_.back());
      } else {
        if (difftime(valid_models_.back()->GetLastUpdateTime(), last_update_time) < 0) {
          valid_models_.back()->SetLastUpdateTime(last_update_time);
        }
      }
    } else {
      // not support
    }

    if (stop_poll == true) {
      break;
    }
  }
}

VersionController::VersionController(int32_t poll_model_wait_seconds, const std::string &models_path,
                                     const std::string &model_name)
    : version_control_strategy_(kLastest),
      poll_model_wait_seconds_(poll_model_wait_seconds),
      models_path_(models_path),
      model_name_(model_name) {}

void StopPollModelPeriodic() { stop_poll = true; }

VersionController::~VersionController() {
  StopPollModelPeriodic();
  if (poll_model_thread_.joinable()) {
    poll_model_thread_.join();
  }
}

Status VersionController::Run() {
  Status ret;
  ret = CreateInitModels();
  if (ret != SUCCESS) {
    return ret;
  }
  // disable periodic check
  // StartPollModelPeriodic();
  return SUCCESS;
}

Status VersionController::CreateInitModels() {
  if (!DirOrFileExist(models_path_)) {
    MS_LOG(ERROR) << "Model Path Not Exist!" << std::endl;
    return FAILED;
  }
  std::vector<std::string> SubDirs = GetAllSubDirs(models_path_);
  if (version_control_strategy_ == kLastest) {
    std::string model_version = GetVersionFromPath(models_path_);
    time_t last_update_time = GetModifyTime(models_path_);
    MindSporeModelPtr model_ptr =
      std::make_shared<MindSporeModel>(model_name_, models_path_, model_version, last_update_time);
    valid_models_.emplace_back(model_ptr);
  } else {
    for (auto &dir : SubDirs) {
      std::string model_version = GetVersionFromPath(dir);
      time_t last_update_time = GetModifyTime(dir);
      MindSporeModelPtr model_ptr = std::make_shared<MindSporeModel>(model_name_, dir, model_version, last_update_time);
      valid_models_.emplace_back(model_ptr);
    }
  }
  if (valid_models_.empty()) {
    MS_LOG(ERROR) << "There is no valid model for serving";
    return FAILED;
  }
  auto ret = Session::Instance().Warmup(valid_models_.back());
  return ret;
}

void VersionController::StartPollModelPeriodic() {
  poll_model_thread_ = std::thread(
    PeriodicFunction(poll_model_wait_seconds_, models_path_, version_control_strategy_, std::ref(valid_models_)));
}

void VersionController::StopPollModelPeriodic() {}
}  // namespace serving
}  // namespace mindspore
