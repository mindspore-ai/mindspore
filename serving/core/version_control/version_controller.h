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
#ifndef MINDSPORE_SERVING_VERSOIN_CONTROLLER_H_
#define MINDSPORE_SERVING_VERSOIN_CONTROLLER_H_

#include <string>
#include <vector>
#include <thread>
#include "./model.h"
#include "util/status.h"

namespace mindspore {
namespace serving {
class VersionController {
 public:
  enum VersionControllerStrategy { kLastest = 0, kMulti = 1 };

  VersionController(int32_t poll_model_wait_seconds, const std::string &models_path, const std::string &model_name);
  ~VersionController();
  Status Run();
  void StartPollModelPeriodic();
  void StopPollModelPeriodic();

 private:
  Status CreateInitModels();

 private:
  VersionControllerStrategy version_control_strategy_;
  std::vector<MindSporeModelPtr> valid_models_;
  int32_t poll_model_wait_seconds_;
  std::thread poll_model_thread_;
  std::string models_path_;
  std::string model_name_;
};

class PeriodicFunction {
 public:
  PeriodicFunction(int32_t poll_model_wait_seconds, const std::string &models_path,
                   VersionController::VersionControllerStrategy version_control_strategy,
                   const std::vector<MindSporeModelPtr> &valid_models)
      : poll_model_wait_seconds_(poll_model_wait_seconds),
        models_path_(models_path),
        version_control_strategy_(version_control_strategy),
        valid_models_(valid_models) {}
  ~PeriodicFunction() = default;
  void operator()();

 private:
  int32_t poll_model_wait_seconds_;
  std::string models_path_;
  VersionController::VersionControllerStrategy version_control_strategy_;
  std::vector<MindSporeModelPtr> valid_models_;
};

}  // namespace serving
}  // namespace mindspore

#endif  // !MINDSPORE_SERVING_VERSOIN_CONTROLLER_H_
