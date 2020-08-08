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

#ifndef MINDSPORE_CCSRC_DEBUG_ASYNC_DUMP_JSON_PARE_H_
#define MINDSPORE_CCSRC_DEBUG_ASYNC_DUMP_JSON_PARE_H_

#include <string>
#include <map>
#include <mutex>
#include <optional>
#include "nlohmann/json.hpp"
#include "utils/ms_utils.h"

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
  uint32_t op_debug_mode() const { return op_debug_mode_; }
  uint32_t dump_step() const { return dump_step_; }
  void MatchKernel(const std::string &kernel_name);
  void PrintUnusedKernel();

 private:
  DataDumpParser() = default;
  virtual ~DataDumpParser() = default;
  DISABLE_COPY_AND_ASSIGN(DataDumpParser);

  void ResetParam();
  bool IsConfigExist(const nlohmann::json &dump_settings) const;
  bool ParseDumpSetting(const nlohmann::json &dump_settings);
  void CheckDumpMode(uint32_t dump_mode) const;
  void CheckOpDebugMode(uint32_t op_debug_mode) const;

  std::mutex lock_;
  bool enable_{false};
  std::string net_name_;
  uint32_t op_debug_mode_{0};
  uint32_t dump_mode_{0};
  uint32_t dump_step_{0};
  std::map<std::string, uint32_t> kernel_map_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_ASYNC_DUMP_JSON_PARE_H_
