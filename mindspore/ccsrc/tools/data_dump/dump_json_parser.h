/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DUMP_JSON_PARSER_H_
#define MINDSPORE_MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DUMP_JSON_PARSER_H_

#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <string>
#include <vector>

#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "nlohmann/json.hpp"
#include "tools/visible.h"
#include "utils/ms_utils.h"

namespace mindspore {
void CheckJsonUnsignedType(const nlohmann::json &content, const std::string &key);
void CheckJsonStringType(const nlohmann::json &content, const std::string &key);
void CheckJsonArrayType(const nlohmann::json &content, const std::string &key);

class TOOLS_EXPORT DumpJsonParser {
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
  void CopyMSCfgJsonToDir(uint32_t rank_id);
  bool NeedDump(const std::string &op_full_name);
  void MatchKernel(const std::string &kernel_name);
  void PrintUnusedKernel();
  bool IsStatisticDump() const;
  bool IsTensorDump() const;
  bool IsFullDump() const;
  bool IsNpyFormat() const;
  bool IsDumpIter(uint32_t iteration) const;
  bool async_dump_enabled() const { return async_dump_enabled_; }
  bool e2e_dump_enabled() const { return e2e_dump_enabled_; }
  bool e2e_sync_dump_enabled() const { return e2e_sync_dump_enabled_; }
  std::vector<std::string> statistic_category() const { return statistic_category_; }
  uint32_t dump_mode() const { return dump_mode_; }
  uint32_t overflow_number() const { return overflow_number_; }
  std::string path() const { return path_; }
  std::string saved_data() const { return saved_data_; }
  std::string iteration_string() const { return iteration_; }
  std::string net_name() const { return net_name_; }
  uint32_t op_debug_mode() const { return op_debug_mode_; }
  bool trans_flag() const { return trans_flag_; }
  bool save_args_flag() const { return save_args_flag_; }
  uint32_t sample_mode() const { return sample_mode_; }
  uint32_t sample_num() const { return sample_num_; }
  uint32_t cur_dump_iter() const;
  uint32_t input_output() const { return input_output_; }
  void UpdateDumpIter() { ++cur_dump_iter_; }
  void UpdateDumpIter(int cur_step_count);
  void UpdateUserDumpStep(const uint32_t step);
  bool GetDatasetSink() { return is_dataset_sink_; }
  void SetDatasetSink(bool is_dataset_sink) { is_dataset_sink_ = is_dataset_sink; }
  bool FileFormatIsNpy() const { return file_format_ == JsonFileFormat::FORMAT_NPY; }
  bool GetIterDumpFlag() const;
  bool DumpEnabledForIter() const;
  bool InputNeedDump() const;
  bool OutputNeedDump() const;
  void GetCellDumpFlag(const session::KernelGraph &kernel_graph);
  void UpdateNeedDumpKernels(const session::KernelGraph &kernel_graph);
  bool IsDumpEnabled();
  bool IsDeviceCalcStats() const;
  void CheckE2eSetting();
  bool IsCallbackRegistered() { return dumpdatacallback_registered_; }
  void SetCallbackRegistered() { dumpdatacallback_registered_ = true; }
  void SetInitialIteration(uint32_t initial_iteration);

  void ClearGraph() { graphs_.clear(); }
  void SaveGraph(session::KernelGraph *graph) { (void)graphs_.emplace_back(graph); }
  const std::vector<session::KernelGraph *> &graphs() const { return graphs_; }
  enum JsonDumpMode { DUMP_ALL = 0, DUMP_KERNEL = 1, DUMP_KERNELS_WITH_FLAG = 2 };
  enum JsonFileFormat { FORMAT_NPY = 0, FORMAT_BIN = 1 };
  enum JsonInputOutput { DUMP_BOTH = 0, DUMP_INPUT = 1, DUMP_OUTPUT = 2 };
  enum JosonOpDebugMode {
    DUMP_WHOLE = 0,
    DUMP_AICORE_OVERFLOW = 1,
    DUMP_ATOMIC_OVERFLOW = 2,
    DUMP_BOTH_OVERFLOW = 3,
    DUMP_LITE_EXCEPTION = 4
  };
  enum JosonSampleMode { DUMP_NORMAL = 0, DUMP_HEAD_AND_TAIL = 1 };
  bool IsDeviceStatHighPrecisionMode() const;
  nlohmann::json GetKernelsJson() { return kernels_json_; }
  std::map<std::string, std::regex> GetKernelRegs() { return kernel_regs_; }
  std::map<std::string, uint32_t> GetKernelStrs() { return kernel_strings_; }

 private:
  DumpJsonParser() = default;
  DISABLE_COPY_AND_ASSIGN(DumpJsonParser)

  inline static std::shared_ptr<DumpJsonParser> instance_ = nullptr;
  inline static std::once_flag instance_mutex_;

  inline static std::mutex lock_;
  bool async_dump_enabled_{false};
  bool e2e_dump_enabled_{false};
  bool e2e_sync_dump_enabled_{false};
  bool is_dataset_sink_{false};
  bool dumpdatacallback_registered_{false};
  uint32_t dump_mode_{0};
  uint32_t overflow_number_{0};
  std::string path_;
  std::string net_name_;
  std::string saved_data_;
  std::string iteration_;
  uint32_t input_output_{0};
  std::map<std::string, uint32_t> kernels_;
  std::map<std::string, uint32_t> kernel_types_;
  std::map<std::string, std::regex> kernel_regs_;
  std::map<std::string, uint32_t> kernel_strings_;
  std::vector<std::string> cell_dump_kernels_;
  std::set<std::string> hccl_input_kernels_;
  std::set<uint32_t> support_devices_;
  uint32_t op_debug_mode_{0};
  JsonFileFormat file_format_{FORMAT_BIN};
  bool trans_flag_{false};
  bool save_args_flag_{false};
  uint32_t sample_mode_{0};
  uint32_t sample_num_{100};
  uint32_t cur_dump_iter_{0};
  uint32_t cur_user_dump_iter_{0};
  uint32_t initial_dump_iter_{0};
  bool already_parsed_{false};
  bool dump_user_step_flag_{false};
  std::string stat_calc_mode_{"host"};
  std::string device_stat_precision_mode_{"high"};
  nlohmann::json kernels_json_ = nlohmann::json::array();
  std::vector<std::string> statistic_category_;

  // Save graphs for dump.
  std::vector<session::KernelGraph *> graphs_;

  void ParseCommonDumpSetting(const nlohmann::json &content);
  void ParseE2eDumpSetting(const nlohmann::json &content);

  static auto CheckJsonKeyExist(const nlohmann::json &content, const std::string &key);
  static bool CheckSelectableKeyExist(const nlohmann::json &content, const std::string &key);

  void ParseDumpMode(const nlohmann::json &content);
  void ParseDumpPath(const nlohmann::json &content);
  void ParseNetName(const nlohmann::json &content);
  void ParseSavedData(const nlohmann::json &content);
  void ParseIteration(const nlohmann::json &content);
  void ParseInitialIteration(const nlohmann::json &content);
  void ParseInputOutput(const nlohmann::json &content);
  void ParseKernels(const nlohmann::json &content);
  void ParseSupportDevice(const nlohmann::json &content);
  bool ParseEnable(const nlohmann::json &content) const;
  void ParseSampleMode(const nlohmann::json &content);
  void ParseSampleNum(const nlohmann::json &content);
  void ParseOpDebugMode(const nlohmann::json &content);
  void ParseFileFormat(const nlohmann::json &content);
  void ParseStatCalcMode(const nlohmann::json &content);
  void ParseDeviceStatPrecisionMode(const nlohmann::json &content);
  void ParseOverflowNumber(const nlohmann::json &content);
  void ParseE2eSyncDumpEnable(const nlohmann::json &content);

  void JudgeDumpEnabled();
  void JsonConfigToString();
  void CheckStatCalcModeVaild();
  void ParseStatisticCategory(const nlohmann::json &content);
  void CheckOverflowSetting();
  void HandleDeviceStatisticCategory(const nlohmann::json::const_iterator user_statistics,
                                     std::string *unsupported_items);
  void HandleHostStatisticCategory(const nlohmann::json::const_iterator user_statistics,
                                   std::string *unsupported_items);
};
bool InputNeedDump();
void DumpJsonParserParse();
void CopyDumpJsonToDir(uint32_t);
void CopyMSCfgJsonToDir(uint32_t);
bool E2eDumpEnabled();
bool AsyncDumpEnabled();
std::string DumpJsonParserPath();
void DumpJsonParserFinalize();
void UpdateNeedDumpKernels(const session::KernelGraph &);
bool DumpJsonParserDumpToFile(const std::string &, const void *, size_t, const ShapeVector &, TypeId);
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DUMP_JSON_PARSER_H_
