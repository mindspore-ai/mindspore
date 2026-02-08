/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PS_CORE_FILE_CONFIGURATION_H_
#define MINDSPORE_CCSRC_PS_CORE_FILE_CONFIGURATION_H_

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <nlohmann/json.hpp>

#include "include/cluster/rpc/constants.h"
#include "utils/log_adapter.h"
#include "cluster/rpc/core/comm_util.h"
#include "cluster/rpc/core/configuration.h"

namespace mindspore {
namespace ps {
namespace core {
// File storage persistent information.
// for example
// {
//   "scheduler_ip": "127.0.0.1",
//   "scheduler_port": 1,
//   "worker_num": 8,
//   "server_num": 16,
//   "total_node_num": 16
// }
class CLUSTER_EXPORT FileConfiguration : public Configuration {
 public:
  explicit FileConfiguration(const std::string &path) : file_path_(path), is_initialized_(false) {}
  ~FileConfiguration() = default;

  bool Initialize() override;

  bool IsInitialized() const override;

  std::string Get(const std::string &key, const std::string &defaultvalue) const override;

  std::string GetString(const std::string &key, const std::string &defaultvalue) const override;

  std::vector<nlohmann::json> GetVector(const std::string &key) const override;

  int64_t GetInt(const std::string &key, int64_t default_value) const override;

  template <typename T>
  T GetJsonValue(const nlohmann::json &json, const std::string &key) {
    auto obj_json = json.find(key);
    if (obj_json != json.end()) {
      try {
        T value = obj_json.value();
        return value;
      } catch (std::exception &e) {
        MS_LOG(ERROR) << "Get Json Value Error, error info: " << e.what();
        MS_LOG(EXCEPTION) << "Get Json Value Error, target type: " << typeid(T).name() << ", key: [" << key << "]"
                          << ", json dump: " << json.dump();
      }
    } else {
      MS_LOG(EXCEPTION) << "Get Json Value Error, can not find key [" << key << "], json dump: " << json.dump();
    }
  }

  template <typename T>
  T GetValue(const std::string &key) const {
    if (!js.contains(key)) {
      MS_LOG(EXCEPTION) << "The key:" << key << " is not exist.";
    }

    return GetJsonValue<T>(js, key);
  }

  void Put(const std::string &key, const std::string &value) override;

  template <typename T>
  void PutValue(const std::string &key, const T &value) {
    std::ofstream output_file(file_path_);
    js[key] = value;
    output_file << js.dump();
    output_file.close();
  }

  bool Exists(const std::string &key) const override;

  void PersistFile(const core::ClusterConfig &clusterConfig) const override {}

  void PersistNodes(const core::ClusterConfig &clusterConfig) const override {}

  std::string file_path() const override;

 private:
  // The path of the configuration file.
  std::string file_path_;

  nlohmann::json js;

  std::atomic<bool> is_initialized_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_FILE_CONFIGURATION_H_
