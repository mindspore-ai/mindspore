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

#ifndef MINDSPORE_MINDSPORE_CCSRC_DEBUG_DUMP_JSON_PARSER_H_
#define MINDSPORE_MINDSPORE_CCSRC_DEBUG_DUMP_JSON_PARSER_H_

#include <string>
#include <map>
#include <set>
#include <mutex>
#include "nlohmann/json.hpp"
#include "utils/ms_utils.h"
#include "backend/session/kernel_graph.h"
namespace mindspore {
class DumpJsonParser {
 public:
  static DumpJsonParser &GetInstance() {
    static DumpJsonParser instance;
    return instance;
  }

  void Parse();
  static bool DumpToFile(const std::string &filename, const void *data, size_t len);
  void CopyJsonToDir();
  bool NeedDump(const std::string &op_full_name) const;
  void MatchKernel(const std::string &kernel_name);
  void PrintUnusedKernel();

  bool async_dump_enabled() const { return async_dump_enabled_; }
  bool e2e_dump_enabled() const { return e2e_dump_enabled_; }
  uint32_t dump_mode() const { return dump_mode_; }
  std::string path() const { return path_; }
  std::string net_name() const { return net_name_; }
  uint32_t iteration() const { return iteration_; }
  uint32_t input_output() const { return input_output_; }
  uint32_t op_debug_mode() const { return op_debug_mode_; }
  bool trans_flag() const { return trans_flag_; }
  uint32_t cur_dump_iter() const { return cur_dump_iter_; }
  void UpdateDumpIter() { ++cur_dump_iter_; }
  bool GetIterDumpFlag();
  bool InputNeedDump() const;
  bool OutputNeedDump() const;
  std::string GetOpOverflowBinPath(uint32_t graph_id, uint32_t device_id) const;
  void UpdateNeedDumpKernels(NotNull<const session::KernelGraph *> kernel_graph);

 private:
  DumpJsonParser() = default;
  ~DumpJsonParser() = default;
  DISABLE_COPY_AND_ASSIGN(DumpJsonParser)

  std::mutex lock_;
  bool async_dump_enabled_{false};
  bool e2e_dump_enabled_{false};
  uint32_t dump_mode_{0};
  std::string path_;
  std::string net_name_;
  uint32_t iteration_{0};
  uint32_t input_output_{0};
  std::map<std::string, uint32_t> kernels_;
  std::set<uint32_t> support_devices_;
  uint32_t op_debug_mode_{0};
  bool trans_flag_{false};
  uint32_t cur_dump_iter_{0};
  bool already_parsed_{false};

  void ParseCommonDumpSetting(const nlohmann::json &content);
  void ParseAsyncDumpSetting(const nlohmann::json &content);
  void ParseE2eDumpSetting(const nlohmann::json &content);
  bool IsDumpEnabled();

  auto CheckJsonKeyExist(const nlohmann::json &content, const std::string &key);

  void ParseDumpMode(const nlohmann::json &content);
  void ParseDumpPath(const nlohmann::json &content);
  void ParseNetName(const nlohmann::json &content);
  void ParseIteration(const nlohmann::json &content);
  void ParseInputOutput(const nlohmann::json &content);
  void ParseKernels(const nlohmann::json &content);
  void ParseSupportDevice(const nlohmann::json &content);
  bool ParseEnable(const nlohmann::json &content);
  void ParseOpDebugMode(const nlohmann::json &content);

  void JudgeDumpEnabled();
  void JsonConfigToString();
};
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_DEBUG_DUMP_JSON_PARSER_H_
