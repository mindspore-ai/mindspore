/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include <vector>
#include <memory>
#include "nlohmann/json.hpp"
#include "utils/ms_utils.h"
#include "backend/common/session/kernel_graph.h"
#include "include/backend/visible.h"

namespace mindspore {
class BACKEND_EXPORT DumpJsonParser {
 public:
  static DumpJsonParser &GetInstance() {
    std::call_once(instance_mutex_, []() {
      if (instance_ == nullptr) {
        instance_ = std::shared_ptr<DumpJsonParser>(new DumpJsonParser);
      }
    });
    return *instance_;
  }
  static void Finalize() { instance_ = nullptr; }

  ~DumpJsonParser() = default;
  void Parse();
  static bool DumpToFile(const std::string &filename, const void *data, size_t len, const ShapeVector &shape,
                         TypeId type);
  void CopyDumpJsonToDir(uint32_t rank_id);
  void CopyHcclJsonToDir(uint32_t rank_id);
  void CopyMSCfgJsonToDir(uint32_t rank_id);
  bool NeedDump(const std::string &op_full_name) const;
  void MatchKernel(const std::string &kernel_name);
  void PrintUnusedKernel();
  bool IsStatisticDump() const;
  bool IsTensorDump() const;
  bool IsFullDump() const;
  bool IsNpyFormat() const;
  bool IsDumpIter(uint32_t iteration) const;
  std::string dump_layer() const { return dump_layer_; }
  bool async_dump_enabled() const { return async_dump_enabled_; }
  bool e2e_dump_enabled() const { return e2e_dump_enabled_; }
  uint32_t dump_mode() const { return dump_mode_; }
  std::string path() const { return path_; }
  std::string saved_data() const { return saved_data_; }
  std::string iteration_string() const { return iteration_; }
  std::string net_name() const { return net_name_; }
  uint32_t op_debug_mode() const { return op_debug_mode_; }
  bool trans_flag() const { return trans_flag_; }
  uint32_t cur_dump_iter() const { return cur_dump_iter_; }
  uint32_t input_output() const { return input_output_; }
  void UpdateDumpIter() { ++cur_dump_iter_; }
  bool FileFormatIsNpy() const { return file_format_ == JsonFileFormat::FORMAT_NPY; }
  bool GetIterDumpFlag() const;
  bool DumpEnabledForIter() const;
  bool InputNeedDump() const;
  bool OutputNeedDump() const;
  std::string GetOpOverflowBinPath(uint32_t graph_id) const;
  void GetCellDumpFlag(const session::KernelGraph &kernel_graph);
  void UpdateNeedDumpKernels(const session::KernelGraph &kernel_graph);
  bool IsDumpEnabled();

  void ClearGraph() { graphs_.clear(); }
  void SaveGraph(session::KernelGraph *graph) { (void)graphs_.emplace_back(graph); }
  const std::vector<session::KernelGraph *> &graphs() const { return graphs_; }
  enum JsonDumpMode { DUMP_ALL = 0, DUMP_KERNEL = 1, DUMP_KERNELS_WITH_FLAG = 2 };
  enum JsonFileFormat { FORMAT_NPY = 0, FORMAT_BIN = 1 };

 private:
  DumpJsonParser() = default;
  DISABLE_COPY_AND_ASSIGN(DumpJsonParser)

  inline static std::shared_ptr<DumpJsonParser> instance_ = nullptr;
  inline static std::once_flag instance_mutex_;

  std::mutex lock_;
  bool async_dump_enabled_{false};
  bool e2e_dump_enabled_{false};
  uint32_t dump_mode_{0};
  std::string path_;
  std::string net_name_;
  std::string saved_data_;
  std::string iteration_;
  uint32_t input_output_{0};
  std::map<std::string, uint32_t> kernels_;
  std::map<std::string, uint32_t> kernel_types_;
  std::vector<std::string> cell_dump_kernels_;
  std::set<uint32_t> support_devices_;
  uint32_t op_debug_mode_{0};
  JsonFileFormat file_format_{FORMAT_BIN};
  bool trans_flag_{false};
  uint32_t cur_dump_iter_{0};
  bool already_parsed_{false};
  bool dump_enabled_warning_printed_{false};
  std::string dump_layer_;

  // Save graphs for dump.
  std::vector<session::KernelGraph *> graphs_;

  void ParseCommonDumpSetting(const nlohmann::json &content);
  void ParseE2eDumpSetting(const nlohmann::json &content);

  static auto CheckJsonKeyExist(const nlohmann::json &content, const std::string &key);

  void ParseDumpMode(const nlohmann::json &content);
  void ParseDumpPath(const nlohmann::json &content);
  void ParseNetName(const nlohmann::json &content);
  void ParseSavedData(const nlohmann::json &content);
  void ParseIteration(const nlohmann::json &content);
  void ParseInputOutput(const nlohmann::json &content);
  void ParseKernels(const nlohmann::json &content);
  void ParseSupportDevice(const nlohmann::json &content);
  bool ParseEnable(const nlohmann::json &content) const;
  void ParseOpDebugMode(const nlohmann::json &content);
  void ParseFileFormat(const nlohmann::json &content);

  void JudgeDumpEnabled();
  void JsonConfigToString();
};
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_DEBUG_DUMP_JSON_PARSER_H_
