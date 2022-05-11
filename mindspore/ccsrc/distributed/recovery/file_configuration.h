/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_RECOVERY_FILE_CONFIGURATION_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_RECOVERY_FILE_CONFIGURATION_H_

#include <string>
#include <nlohmann/json.hpp>
#include "distributed/recovery/configuration.h"

namespace mindspore {
namespace distributed {
namespace recovery {
// Local file for saving and restore metadata.
class FileConfiguration : public Configuration {
 public:
  explicit FileConfiguration(const std::string &path) : file_(path) {}
  ~FileConfiguration() = default;

  bool Initialize() override;

  std::string Get(const std::string &key, const std::string &defaultvalue) const override;

  void Put(const std::string &key, const std::string &value) override;

  bool Exists(const std::string &key) const override;

  bool Empty() const override;

  bool Flush() override;

 private:
  // The full path of the local configuration file.
  std::string file_;

  // All the key-value pairs managed by this configuration are stored in json format.
  nlohmann::json values_;
};
}  // namespace recovery
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_RECOVERY_FILE_CONFIGURATION_H_
