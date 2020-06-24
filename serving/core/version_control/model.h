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
#ifndef MINDSPORE_SERVING_MODEL_H_
#define MINDSPORE_SERVING_MODEL_H_

#include <string>
#include <ctime>
#include <memory>

namespace mindspore {
namespace serving {
class MindSporeModel {
 public:
  MindSporeModel(const std::string &model_name, const std::string &model_path, const std::string &model_version,
                 const time_t &last_update_time);
  ~MindSporeModel() = default;
  std::string GetModelName() { return model_name_; }
  std::string GetModelPath() { return model_path_; }
  std::string GetModelVersion() { return model_version_; }
  time_t GetLastUpdateTime() { return last_update_time_; }
  void SetLastUpdateTime(const time_t &last_update_time) { last_update_time_ = last_update_time; }

 private:
  std::string model_name_;
  std::string model_path_;
  std::string model_version_;
  time_t last_update_time_;
};

using MindSporeModelPtr = std::shared_ptr<MindSporeModel>;
}  // namespace serving
}  // namespace mindspore

#endif  // !MINDSPORE_SERVING_MODEL_H_
