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

#ifndef MINDSPORE_MINDSPORE_CCSRC_DEBUG_ASYNC_DUMP_JSON_PARE_H_
#define MINDSPORE_MINDSPORE_CCSRC_DEBUG_ASYNC_DUMP_JSON_PARE_H_

#include <string>
#include <set>
#include <mutex>
#include <optional>
#include "nlohmann/json.hpp"
#include "common/utils.h"

namespace mindspore {
class DataDumpParser {
 public:
  static DataDumpParser &GetInstance() {
    static DataDumpParser instance;
    return instance;
  }
  void ParseDumpConfig();
  bool NeedDump(const std::string &op_full_name) const;
  bool DumpEnabled() const;
  std::optional<std::string> GetDumpPath() const;
  bool enable() const { return enable_; }
  const std::string &net_name() const { return net_name_; }
  uint32_t dump_mode() const { return dump_mode_; }
  uint32_t dump_step() const { return dump_step_; }
  const std::set<std::string> &kernel_set() const { return kernel_set_; }

 private:
  DataDumpParser() = default;
  virtual ~DataDumpParser() = default;
  DISABLE_COPY_AND_ASSIGN(DataDumpParser);

  void ResetParam();
  bool IsConfigExist(const nlohmann::json &dump_settings) const;
  bool ParseDumpSetting(const nlohmann::json &dump_settings);

  std::mutex lock_;
  bool enable_{false};
  std::string net_name_;
  uint32_t dump_mode_{0};
  uint32_t dump_step_{0};
  std::set<std::string> kernel_set_;
};
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_DEBUG_ASYNC_DUMP_JSON_PARE_H_
