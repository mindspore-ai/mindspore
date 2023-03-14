/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_DEBUG_DEBUG_SERVICES_H_
#define MINDSPORE_CCSRC_DEBUG_DEBUG_SERVICES_H_

#ifdef OFFLINE_DBG_MODE
#include "base/float16.h"
#endif

#include <cmath>
#include <vector>
#include <future>
#include <string>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <set>
#include <mutex>
#include <map>
#include <limits>
#include <sstream>
#include <utility>
#include "debug/tensor_load.h"
#include "include/backend/debug/tensor_data.h"

namespace mindspore {
class DebugServices {
 public:
  DebugServices();

  DebugServices(const DebugServices &other);

  DebugServices &operator=(const DebugServices &other);

  ~DebugServices() = default;
  enum File_ATTR_MATCH { START_POS = 0, END_POS = 1, STR_POS = 2 };

  enum CONDITION_TYPE {
    HAS_NAN,
    HAS_INF,
    IS_OVERFLOW,
    MAX_GT,
    MAX_LT,
    MIN_GT,
    MIN_LT,
    MAX_MIN_GT,
    MAX_MIN_LT,
    MEAN_GT,
    MEAN_LT,
    SD_GT,
    SD_LT,
    GENERAL_OVERFLOW,
    INIT,
    TOO_LARGE,
    TOO_SMALL,
    ALL_ZERO,
    CHANGE_TOO_LARGE,
    CHANGE_TOO_SMALL,
    NOT_CHANGED,
    RANGE
  };

  struct condition_t {
    CONDITION_TYPE type;
    float parameter = 0;
  };

  struct parameter_t {
    std::string name;
    bool disabled;
    double_t value;
    bool hit;
    double_t actual_value;
    void Evaluate(double_t actualValue, std::string inequality_type) {
      if (std::isnan(actualValue)) {
        return;
      }

      actual_value = actualValue;
      // if cannot extract inequality type from watchpoint
      // try extract from parameter name
      if (inequality_type.empty()) {
        auto pos = name.find_last_of('_');
        if (pos != std::string::npos) {
          inequality_type = name.substr(pos + 1);
        }
      }

      std::map<std::string, bool> condition_check{{"gt", actual_value > value},
                                                  {"lt", actual_value < value},
                                                  {"ge", actual_value >= value},
                                                  {"le", actual_value <= value}};

      hit = condition_check[inequality_type];
    }
  };
  struct MappedFiles {
    std::vector<std::string> bin_files;
    // key is op_name and value is the vector of matched npy files to that op name.
    std::map<std::string, std::vector<std::string>> npy_files;
  };

  struct DumpFileAttr {
    std::string file_path;
    // name_to_match is the op_name extracted from file name.
    std::string name_to_match;
    std::string time_stamp;
    uint64_t slot = 0;
    bool is_output{false};
  };

  struct ProtoDump {
    bool operator==(const ProtoDump obj) {
      return (origin_node_name == obj.origin_node_name && dump_name == obj.dump_name && is_output == obj.is_output);
    }
    // name_to_match is the op_name between first and second dot in file_name
    std::string origin_node_name;
    std::string dump_name;
    bool is_output{false};
  };

  typedef std::vector<std::vector<int>> partitioned_numbers;
  typedef std::vector<std::vector<std::string>> partitioned_names;
  typedef std::vector<std::vector<std::vector<parameter_t>>> partitioned_parameters;
  typedef std::vector<std::vector<int32_t>> partitioned_error_code;
  typedef std::vector<std::vector<unsigned int>> partitioned_id;
  typedef std::set<std::string> NPYFilePool;
  typedef std::map<std::string, std::vector<std::tuple<std::string, std::string>>> DirMap;
  // key is dump dir path and value is vector of bin files and map of npy files.
  typedef std::map<std::string, DebugServices::MappedFiles> DumpFileMap;
  typedef std::map<std::string, std::vector<DebugServices::DumpFileAttr>> ProcessedNPYFiles;
  // bool shows if preprocess was successful, and DumpFileMap is preprocessed file result
  typedef std::tuple<bool, DumpFileMap> AsyncPreProcessResult;

  struct watchpoint_t {
    unsigned int id;
    condition_t condition;
    std::vector<std::tuple<std::string, bool>> check_node_list;
    std::vector<std::tuple<std::string, std::vector<uint32_t>>> check_node_device_list;
    std::vector<std::tuple<std::string, std::vector<uint32_t>>> check_node_graph_list;
    std::vector<parameter_t> parameter_list;
    size_t location = 0;

    std::string FindQualifiedTensorName(const std::string &tensor_name, unsigned const int &tensor_device_id,
                                        unsigned const int &tensor_root_graph_id) const {
      size_t indx = 0;
      for (auto check_node : check_node_list) {
        std::string w_name = std::get<0>(check_node);
        bool w_type = std::get<1>(check_node);
        auto found = w_name.find_last_of('/');
        bool check_tensor_name = found != std::string::npos && w_name.substr(found + 1) == tensor_name;
        bool check_node_name =
          (w_type && (tensor_name == w_name || w_name == "*")) || (!w_type && tensor_name == w_name);
        if (check_tensor_name || check_node_name) {
          // online debugger only support single card
          if (check_node_device_list.empty()) {
            return w_name;
          }
          auto device_vec = std::get<1>(check_node_device_list[indx]);
          auto root_graph_vec = std::get<1>(check_node_graph_list[indx]);
          auto iter1 = std::find(device_vec.begin(), device_vec.end(), tensor_device_id);
          auto iter2 = std::find(root_graph_vec.begin(), root_graph_vec.end(), tensor_root_graph_id);
          if (iter1 != device_vec.end() && iter2 != root_graph_vec.end()) {
            return w_name;
          }
        }
        indx++;
      }
      return {};
    }

    bool is_gt_wp() const {
      return condition.type == MAX_GT || condition.type == MIN_GT || condition.type == MEAN_GT ||
             condition.type == SD_GT || condition.type == MAX_MIN_GT;
    }

    bool is_lt_wp() const {
      return condition.type == MAX_LT || condition.type == MIN_LT || condition.type == MEAN_LT ||
             condition.type == SD_LT || condition.type == MAX_MIN_LT;
    }

    // mean or sd related condition set
    bool mean_sd_enabled() const {
      return condition.type == MEAN_LT || condition.type == MEAN_GT || condition.type == SD_LT ||
             condition.type == SD_GT || (condition.type == TOO_LARGE && !parameter_list[3].disabled) ||
             (condition.type == TOO_SMALL && !parameter_list[3].disabled);
    }
    bool abs_mean_enabled() const {
      return (condition.type == TOO_LARGE && !parameter_list[0].disabled) ||
             (condition.type == TOO_SMALL && !parameter_list[0].disabled);
    }

    bool tensor_update_ratio_mean_enabled() const {
      return condition.type == CHANGE_TOO_LARGE || condition.type == CHANGE_TOO_SMALL;
    }
    bool allclose_enabled() const { return condition.type == NOT_CHANGED; }

    bool range_enabled() const {
      return condition.type == RANGE && (!parameter_list[0].disabled || !parameter_list[1].disabled);
    }

    bool change_condition() const {
      return condition.type == CHANGE_TOO_LARGE || condition.type == CHANGE_TOO_SMALL || condition.type == NOT_CHANGED;
    }
  };

  struct TensorBase {
    TensorBase(uint64_t data_size, int dtype, const std::vector<int64_t> &shape)
        : data_size(data_size), dtype(dtype), shape(shape) {}
    TensorBase() = default;
    uint64_t data_size = 0;
    int dtype = 0;
    std::vector<int64_t> shape;
  };

  struct TensorStat {
    TensorStat(uint64_t data_size, int dtype, const std::vector<int64_t> &shape, bool is_bool, double max_value,
               double min_value, double avg_value, uint64_t count, uint64_t neg_zero_count, uint64_t pos_zero_count,
               uint64_t nan_count, uint64_t neg_inf_count, uint64_t pos_inf_count, uint64_t zero_count)
        : data_size(data_size),
          dtype(dtype),
          shape(shape),
          is_bool(is_bool),
          max_value(max_value),
          min_value(min_value),
          avg_value(avg_value),
          count(count),
          neg_zero_count(neg_zero_count),
          pos_zero_count(pos_zero_count),
          nan_count(nan_count),
          neg_inf_count(neg_inf_count),
          pos_inf_count(pos_inf_count),
          zero_count(zero_count) {}

    TensorStat() = default;

    uint64_t data_size = 0;
    int dtype = 0;
    std::vector<int64_t> shape;
    bool is_bool = false;
    double max_value = std::numeric_limits<double>::lowest();
    double min_value = std::numeric_limits<double>::max();
    double avg_value = 0.0;
    uint64_t count = 0;
    uint64_t neg_zero_count = 0;
    uint64_t pos_zero_count = 0;
    uint64_t nan_count = 0;
    uint64_t neg_inf_count = 0;
    uint64_t pos_inf_count = 0;
    uint64_t zero_count = 0;
  };

  struct ChunkData {
    partitioned_names chunk_names;
    partitioned_names chunk_slots;
    partitioned_numbers chunk_conditions;
    partitioned_id chunk_watchpoint_id;
    partitioned_parameters chunk_parameters;
    partitioned_error_code chunk_error_codes;
    partitioned_numbers chunk_exec_orders;
    partitioned_id chunk_device_id;
    partitioned_id chunk_root_graph_id;
    std::vector<uint64_t> chunk_tensor_byte_size;
    partitioned_names chunk_time_stamp;
  };

  static TensorStat GetTensorStatistics(const std::shared_ptr<TensorData> &tensor);

  void AddWatchpoint(
    int id, int watch_condition, float parameter, const std::vector<std::tuple<std::string, bool>> &check_node_list,
    const std::vector<parameter_t> &parameter_list,
    const std::vector<std::tuple<std::string, std::vector<uint32_t>>> *check_node_device_list = nullptr,
    const std::vector<std::tuple<std::string, std::vector<uint32_t>>> *check_node_graph_list = nullptr);

  void RemoveWatchpoint(unsigned int id);

#ifdef OFFLINE_DBG_MODE
  void CheckOutofMemoryandNoValue(const bool no_mem_to_read, const bool error_on_no_value,
                                  const std::vector<watchpoint_t> watchpoints_to_check, const int chunk_id,
                                  ChunkData *chunk_data, std::vector<unsigned int> *const device_id,
                                  std::vector<unsigned int> *const root_graph_id, const int exec_order,
                                  const std::string time_stamp, const std::string &qualified_tensor_name,
                                  const std::string &tensor_slot, const unsigned int device_id_val,
                                  const unsigned int root_graph_id_val,
                                  const std::vector<parameter_t> &parameter_list) const;
#endif

  const void *PreparePrevTensor(uint64_t *prev_num_elements, const std::string &tensor_name);

  void CheckHistoryErrorCode(int *error_code, bool history_not_found) const;

  void CheckWatchpointsForTensor(ChunkData *chunk_data, ProcessedNPYFiles *const processed_npy_files,
                                 std::vector<std::shared_ptr<TensorData>> *const tensor_list, int begin, int end,
                                 int chunk_id, const bool init_dbg_suspend, const bool step_end, const bool recheck,
                                 std::vector<unsigned int> *device_id, std::vector<unsigned int> *root_graph_id,
                                 bool error_on_no_value = false);

  void GetOverflowTaskStreamId(const std::string &overflow_bin_path,
                               std::vector<std::pair<uint64_t, uint64_t>> *task_stream_hits) const;

  void GetTaskStreamIdNodeMap(const std::string &tensor_path,
                              std::map<std::pair<uint64_t, uint64_t>, std::string> *task_stream_to_opnames) const;

  void AddOpOverflowOpNames(const std::string &overflow_bin_path, const std::string &tensors_path,
                            std::vector<std::string> *op_names) const;

  void CheckWatchpoints(std::vector<std::string> *name, std::vector<std::string> *slot, std::vector<int> *condition,
                        std::vector<unsigned int> *const watchpoint_id,
                        std::vector<std::vector<parameter_t>> *parameters, std::vector<int32_t> *error_code,
                        ProcessedNPYFiles *const processed_npy_files,
                        std::vector<std::shared_ptr<TensorData>> *tensor_list, bool init_dbg_suspend,
                        const bool step_end, const bool recheck, std::vector<unsigned int> *device_id = nullptr,
                        std::vector<unsigned int> *root_graph_id = nullptr, bool error_on_no_value = false);

  void SortWatchpointsInfo(std::vector<std::future<void>> *const tensor_future_vec, std::vector<int> *exec_order,
                           std::vector<std::string> *time_stamps, uint64_t *tensor_list_byte_size,
                           std::vector<std::string> *name, std::vector<std::string> *slot, std::vector<int> *condition,
                           std::vector<unsigned int> *const watchpoint_id,
                           std::vector<std::vector<parameter_t>> *parameters, std::vector<int32_t> *error_codes,
                           ChunkData *chunk_data, std::vector<unsigned int> *device_id,
                           std::vector<unsigned int> *root_graph_id) const;
#ifdef OFFLINE_DBG_MODE
  void SetTensorToNotInUse(const std::shared_ptr<TensorData> &tensor, const void *previous_tensor_ptr);
#endif

  void AddWatchPointsToCheck(bool init_dbg_suspend, bool step_end, bool recheck,
                             const std::shared_ptr<TensorData> &tensor, bool *previous_iter_tensor_needed,
                             std::string *qualified_tensor_name, std::vector<watchpoint_t> *watchpoints_to_check);

  void SetCheckWatchpointsResult(const int chunk_id, ChunkData *chunk_data, std::vector<unsigned int> *device_id,
                                 std::vector<unsigned int> *root_graph_id, const int exec_order,
                                 const std::string time_stamp, const std::string &qualified_tensor_name,
                                 const std::string &tensor_slot, const watchpoint_t &wp,
                                 const unsigned int device_id_val, const unsigned int root_graph_id_val,
                                 const std::vector<parameter_t> &parameter_list, const int32_t error_code) const;
#ifdef OFFLINE_DBG_MODE
  void AddToTensorData(const std::string &backend_name, const std::string &time_stamp, const std::size_t slot,
                       const unsigned int iteration, const unsigned int device_id, const unsigned int root_graph_id,
                       const bool is_output, const std::size_t data_size, const std::string &type_name,
                       const std::vector<int64_t> &shape, char *buffer,
                       std::vector<std::shared_ptr<TensorData>> *const result_list);

  void SetPrefixToCheck(std::string *const prefix_dump_file_name, std::string *const slot_string_to_check,
                        std::string *const dump_style_kernel_name, size_t slot, bool is_output);

  void ReadDumpedTensor(std::vector<std::string> backend_name, std::vector<size_t> slot,
                        std::vector<unsigned int> device_id, std::vector<unsigned int> iteration,
                        std::vector<unsigned int> root_graph_id, const std::vector<bool> &is_output,
                        ProcessedNPYFiles *const processed_npy_files,
                        std::vector<std::shared_ptr<TensorData>> *const result_list, bool is_base_request,
                        bool *no_mem_to_read = nullptr);

  void ProcessTensorDataSync(const std::vector<ProtoDump> &proto_to_dump, const std::string &specific_dump_dir,
                             ProcessedNPYFiles processed_npy_files, unsigned int iteration, unsigned int device_id,
                             unsigned int root_graph_id, std::vector<std::shared_ptr<TensorData>> *const tensor_list,
                             bool error_on_no_value = false);

  void ReadFileAndAddToTensor(const bool found, const std::vector<std::string> &matched_paths,
                              const std::vector<std::string> &matched_time_stamps, const std::string &backend_name,
                              const unsigned int device_id, const unsigned int root_graph_id, bool is_output,
                              size_t slot, bool *no_mem_to_read, unsigned int iteration,
                              std::vector<std::shared_ptr<TensorData>> *result_list, bool is_base_request = false);

  void ReadDumpedTensorSync(const std::string &prefix_dump_file_name, const std::string &specific_dump_dir,
                            const std::string &backend_name, size_t slot, unsigned int device_id,
                            unsigned int iteration, unsigned int root_graph_id, const bool &is_output,
                            std::vector<std::shared_ptr<TensorData>> *result_list, bool *no_mem_to_read);

  void ReadDumpedTensorUtils(const std::string &specific_dump_dir, const std::string &prefix_dump_to_check,
                             const std::string &backend_name, size_t slot, unsigned int device_id,
                             unsigned int iteration, unsigned int root_graph_id, bool is_output,
                             const ProcessedNPYFiles &processed_npy_files,
                             std::vector<std::shared_ptr<TensorData>> *result_list, bool *no_mem_to_read,
                             bool is_base_request = false);

  std::vector<std::shared_ptr<TensorData>> ReadNeededDumpedTensors(unsigned int iteration,
                                                                   ProcessedNPYFiles *const processed_npy_files,
                                                                   bool error_on_no_value = false);

  const void *GetPrevTensor(const std::shared_ptr<TensorData> &tensor, bool previous_iter_tensor_needed,
                            uint64_t *prev_num_elements, bool *history_not_found);

  void ReadTensorFromNpy(const std::string &tensor_name, const std::string &file_name, std::string *const tensor_type,
                         std::size_t *const size, std::vector<int64_t> *const shape, char **const data_buffer,
                         bool *no_mem_to_read, bool is_base_request = false);

  AsyncPreProcessResult PreProcessDumpDirAsync(const std::string &specific_dump_dir) const;

  DebugServices::NPYFilePool PreProcessDumpDirSync(const std::string &specific_dump_dir) const;

  ProcessedNPYFiles ProcessNPYFilePool(const NPYFilePool &npy_file_pool) const;

  void ConvertToHostFormat(const DirMap &dir_to_files_map, NPYFilePool *const result_list) const;

  void ProcessConvertToHostFormat(const std::vector<std::string> &files_after_convert_in_dir,
                                  const std::string &dump_key, NPYFilePool *const result_list) const;

  void ConvertReadTensors(std::vector<std::string> backend_name, std::vector<size_t> slot,
                          std::vector<unsigned int> device_id, std::vector<unsigned int> iteration,
                          std::vector<unsigned int> root_graph_id, NPYFilePool *const result_list);

  void ConvertWatchPointNodes(const DumpFileMap &dump_dir_mapped_files, const std::vector<ProtoDump> &proto_dump,
                              const std::string &specific_dump_dir, NPYFilePool *const result_list) const;

  void ProcessConvertList(const DumpFileMap &dump_dir_mapped_files, const std::string &prefix_dump_file_name,
                          const std::string &specific_dump_dir, DirMap *dir_to_files_map,
                          NPYFilePool *const result_list) const;

  void GetTensorDataInfoAsync(const std::vector<ProtoDump> &proto_dump, const std::string &specific_dump_dir,
                              uint32_t iteration, uint32_t device_id, uint32_t root_graph_id,
                              const ProcessedNPYFiles &processed_async_files,
                              std::vector<std::shared_ptr<TensorData>> *const tensor_list);

  void SetGraphsHistory();

  std::vector<uint32_t> GetDumpRankIdList();

  void CheckDumpGraphIdList(std::vector<uint32_t> rank_id_list);

  void ReadGraphsHistory(uint32_t rank_id, uint32_t root_graph_id);

  std::map<std::tuple<uint32_t, uint32_t>, std::vector<std::tuple<std::string, bool>>> GetAllWpNodes();

  void ReadGraphRunIter(std::string file_path, std::tuple<uint32_t, uint32_t> rank_and_graph);

  std::string IterationString(unsigned int iteration) const;
#endif
  void ReadNodesTensors(const std::vector<std::string> &name, std::vector<std::string> *ret_name,
                        std::vector<const char *> *data_ptr, std::vector<ssize_t> *data_size,
                        std::vector<unsigned int> *dtype, std::vector<std::vector<int64_t>> *const shape);

  void SearchNodesTensors(const std::vector<std::string> &name,
                          std::vector<std::tuple<std::string, std::shared_ptr<TensorData>>> *result_list);
#ifndef OFFLINE_DBG_MODE
  bool IsWatchPoint(const std::string &kernel_name, const CNodePtr &kernel = nullptr) const;

  bool IsWatchPointNodeInput(const std::string &w_name, const CNodePtr &kernel) const;

  bool CompareCurrentRootGraph(uint32_t id) const;
#endif

  std::vector<std::shared_ptr<TensorData>> GetTensor() const;

  std::shared_ptr<TensorData> GetTensor(const std::string &tensor_name) const;

  void AddAnalyzedTensorToCache(const bool recheck, const unsigned int id, const std::string &tensor_name);

  void EmptyCurrentTensor();

#ifndef OFFLINE_DBG_MODE
  bool DumpTensorToFile(const std::string &filepath, const std::string &tensor_name, size_t slot) const;
#endif

  bool LoadNewTensor(const std::shared_ptr<TensorData> &tensor, bool keep_prev);

  uint32_t GetPrevIteration(const std::shared_ptr<TensorData> &tensor);

  void ResetLoadedTensors();
#ifndef OFFLINE_DBG_MODE
  std::vector<std::shared_ptr<TensorData>> GetNodeTensor(const CNodePtr &kernel);
#endif

  // Find if any operation overflow happened on a particular node name
  bool CheckOpOverflow(std::string node_name_to_find, unsigned int device_id = 0, unsigned int root_graph_id = 0,
                       unsigned int iteration = 0);

  std::string RemoveKernelGraphPrefix(std::string node_name_to_find) const;

  bool GetTaskIdStreamId(std::string file_name, std::string overflow_file_prefix, uint64_t *const task_id,
                         uint64_t *const stream_id) const;

  bool GetAttrsFromFilename(const std::string &file_name, std::string *const node_name, uint64_t *const task_id,
                            uint64_t *const stream_id) const;

  std::string RealPath(const std::string &input_path) const;

  bool TensorExistsInCurrent(const std::string &tensor_name);

  void MoveTensorCurrentToPrev(const std::string &tensor_name);

  void AppendToCacheEvictQueue(const std::string &tensor_name);

  void SetNetName(std::string net_name);

  std::string GetNetName();

  void SetDumpDir(std::string dump_dir);

  std::string GetDumpDir();

  void SetSyncMode(bool is_sync_mode);

  bool GetSyncMode() const;

  void SetMemLimit(uint64_t max_mem_size);

  void CheckWatchpointProgress(size_t tensor_list_size);

  size_t GetProcessedTensorCount() const { return tensor_processed_count_; }

 private:
  std::mutex lock_;
  std::mutex wp_lock_;
  std::mutex overflow_wp_lock_;

  // to keep track of watchpoints that have been checked already for a tensor in current step
  std::unordered_map<std::string, std::set<int32_t>> wp_id_cache_;
  std::unordered_map<unsigned int, watchpoint_t> watchpoint_table_;
  // key is the iteration path, value is vector of op_names which have overflowed
  std::unordered_map<std::string, std::vector<std::string>> overflow_ops_;
  std::string net_name_;
  std::string dump_dir_;
  // store history of graphs that have been run (rank_id, graph_id)
  std::map<std::tuple<uint32_t, uint32_t>, std::vector<uint32_t>> graphs_run_history_;
  bool is_sync_mode_{false};
  // processed tensors in checkwatchpoint function
  std::atomic<size_t> tensor_processed_count_{0};
  bool wp_progress_enabled_{false};
  std::unique_ptr<std::thread> wp_progress_thread_;
  std::shared_ptr<TensorLoader> tensor_loader_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEBUG_DEBUG_SERVICES_H_
