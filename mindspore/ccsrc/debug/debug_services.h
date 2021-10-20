/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef OFFLINE_DBG_MODE
#define ONLINE_DBG_MODE
#endif

#ifdef OFFLINE_DBG_MODE
#include "base/float16.h"
#endif

#include <math.h>
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
#include "debug/tensor_load.h"
#include "debug/tensor_data.h"

#ifdef ONLINE_DBG_MODE
namespace mindspore {
#endif
class DebugServices {
 public:
  DebugServices();

  DebugServices(const DebugServices &other);

  DebugServices &operator=(const DebugServices &other);

  ~DebugServices() = default;

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

  typedef std::vector<std::vector<int>> partitioned_numbers;
  typedef std::vector<std::vector<std::string>> partitioned_names;
  typedef std::vector<std::vector<std::vector<parameter_t>>> partitioned_parameters;
  typedef std::vector<std::vector<int32_t>> partitioned_error_code;
  typedef std::vector<std::vector<unsigned int>> partitioned_id;

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
      int indx = 0;
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
               double min_value, double avg_value, int count, int neg_zero_count, int pos_zero_count, int nan_count,
               int neg_inf_count, int pos_inf_count, int zero_count)
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
    int count = 0;
    int neg_zero_count = 0;
    int pos_zero_count = 0;
    int nan_count = 0;
    int neg_inf_count = 0;
    int pos_inf_count = 0;
    int zero_count = 0;
  };

  TensorStat GetTensorStatistics(const std::shared_ptr<TensorData> &tensor);

  void AddWatchpoint(
    unsigned int id, unsigned int watch_condition, float parameter,
    const std::vector<std::tuple<std::string, bool>> &check_node_list, const std::vector<parameter_t> &parameter_list,
    const std::vector<std::tuple<std::string, std::vector<uint32_t>>> *check_node_device_list = nullptr,
    const std::vector<std::tuple<std::string, std::vector<uint32_t>>> *check_node_graph_list = nullptr);

  void RemoveWatchpoint(unsigned int id);

#ifdef OFFLINE_DBG_MODE
  void ProcessCheckpointsOutofMemory(
    const bool no_mem_to_read, const std::vector<watchpoint_t> watchpoints_to_check, const int chunk_id,
    partitioned_names *const chunk_names, partitioned_names *const chunk_slots,
    partitioned_numbers *const chunk_conditions, partitioned_id *const chunk_watchpoint_id,
    partitioned_parameters *const chunk_parameters, partitioned_error_code *const chunk_error_codes,
    partitioned_numbers *const chunk_exec_orders, partitioned_names *const chunk_time_stamp,
    partitioned_id *const chunk_device_id, partitioned_id *const chunk_root_graph_id,
    std::vector<unsigned int> *const device_id, std::vector<unsigned int> *const root_graph_id, const int exec_order,
    const std::string time_stamp, const std::string &qualified_tensor_name, const std::string &tensor_slot,
    const unsigned int device_id_val, const unsigned int root_graph_id_val,
    const std::vector<parameter_t> &parameter_list);
#endif

  void CheckWatchpointsForTensor(partitioned_names *chunk_names, partitioned_names *chunk_slots,
                                 partitioned_numbers *chunk_conditions, partitioned_id *const chunk_watchpoint_id,
                                 partitioned_parameters *chunk_parameters, partitioned_error_code *chunk_error_codes,
                                 const std::vector<std::string> &op_overflows,
                                 const std::vector<std::string> &async_file_pool,
                                 partitioned_numbers *chunk_exec_orders,
                                 std::vector<std::shared_ptr<TensorData>> *tensor_list, int begin, int end,
                                 int chunk_id, const bool init_dbg_suspend, const bool step_end, const bool recheck,
                                 partitioned_id *chunk_device_id, partitioned_id *chunk_root_graph_id,
                                 std::vector<uint64_t> *chunk_tensor_byte_size, partitioned_names *chunk_time_stamp,
                                 std::vector<unsigned int> *device_id, std::vector<unsigned int> *root_graph_id);

  void CheckWatchpoints(std::vector<std::string> *name, std::vector<std::string> *slot, std::vector<int> *condition,
                        std::vector<unsigned int> *const watchpoint_id,
                        std::vector<std::vector<parameter_t>> *parameters, std::vector<int32_t> *error_code,
                        const std::vector<std::string> &op_overflows, const std::vector<std::string> &async_file_pool,
                        std::vector<std::shared_ptr<TensorData>> *tensor_list, bool init_dbg_suspend,
                        const bool step_end, const bool recheck, std::vector<unsigned int> *device_id = nullptr,
                        std::vector<unsigned int> *root_graph_id = nullptr);

  void SortWatchpointsInfo(std::vector<std::future<void>> *tensor_future_vec, std::vector<int> *exec_order,
                           std::vector<std::string> *time_stamps, uint64_t *tensor_list_byte_size,
                           std::vector<std::string> *name, std::vector<std::string> *slot, std::vector<int> *condition,
                           std::vector<unsigned int> *const watchpoint_id,
                           std::vector<std::vector<parameter_t>> *parameters, std::vector<int32_t> *error_codes,
                           partitioned_names *chunk_names, partitioned_names *chunk_slots,
                           partitioned_numbers *chunk_conditions, partitioned_id *chunk_watchpoint_id,
                           partitioned_parameters *chunk_parameters, partitioned_error_code *chunk_error_codes,
                           partitioned_numbers *chunk_exec_orders, partitioned_names *chunk_time_stamp,
                           std::vector<uint64_t> *chunk_tensor_byte_size, partitioned_id *chunk_device_id,
                           partitioned_id *chunk_root_graph_id, std::vector<unsigned int> *device_id,
                           std::vector<unsigned int> *root_graph_id);

  void AddWatchPointsToCheck(bool init_dbg_suspend, bool step_end, bool recheck,
                             const std::shared_ptr<TensorData> &tensor, bool *previous_iter_tensor_needed,
                             std::string *qualified_tensor_name, std::vector<watchpoint_t> *watchpoints_to_check);

  void SetCheckWatchpointsResult(const int chunk_id, partitioned_names *chunk_names, partitioned_names *chunk_slots,
                                 partitioned_numbers *chunk_conditions, partitioned_id *chunk_watchpoint_id,
                                 partitioned_parameters *chunk_parameters, partitioned_error_code *chunk_error_codes,
                                 partitioned_numbers *chunk_exec_orders, partitioned_names *chunk_time_stamp,
                                 partitioned_id *chunk_device_id, partitioned_id *chunk_root_graph_id,
                                 std::vector<unsigned int> *device_id, std::vector<unsigned int> *root_graph_id,
                                 const int exec_order, const std::string time_stamp,
                                 const std::string &qualified_tensor_name, const std::string &tensor_slot,
                                 const watchpoint_t &wp, const unsigned int device_id_val,
                                 const unsigned int root_graph_id_val, const std::vector<parameter_t> &parameter_list,
                                 const int32_t error_code);
#ifdef OFFLINE_DBG_MODE
  void AddToTensorData(const std::string &backend_name, const std::string &time_stamp, const std::size_t slot,
                       const unsigned int iteration, const unsigned int device_id, const unsigned int root_graph_id,
                       const bool is_output, const std::size_t data_size, const std::string &type_name,
                       const std::vector<int64_t> &shape, std::vector<char> *buffer,
                       std::vector<std::shared_ptr<TensorData>> *const result_list);

  void SetPrefixToCheck(std::string *const prefix_dump_file_name, std::string *const slot_string_to_check,
                        std::string *const dump_style_kernel_name, size_t slot, bool is_output);

  void ReadDumpedTensor(std::vector<std::string> backend_name, std::vector<size_t> slot,
                        std::vector<unsigned int> device_id, std::vector<unsigned int> iteration,
                        std::vector<unsigned int> root_graph_id, const std::vector<bool> &is_output,
                        const std::vector<std::string> &async_file_pool,
                        std::vector<std::shared_ptr<TensorData>> *const result_list, bool *no_mem_to_read = nullptr);

  void ProcessTensorDataSync(const std::vector<std::tuple<std::string, std::string>> &proto_to_dump,
                             const std::string &abspath, const std::string &specific_dump_dir, unsigned int iteration,
                             unsigned int device_id, unsigned int root_graph_id,
                             std::vector<std::shared_ptr<TensorData>> *const tensor_list);

  void ReadFileAndAddToTensor(const bool found, const std::vector<std::string> &matched_paths,
                              const std::string &backend_name, const unsigned int device_id,
                              const unsigned int root_graph_id, const bool &is_output, size_t slot,
                              bool *no_mem_to_read, unsigned int iteration,
                              std::vector<std::shared_ptr<TensorData>> *result_list);

  void ReadDumpedTensorSync(const std::string &prefix_dump_file_name, const std::string &specific_dump_dir,
                            const std::string &backend_name, size_t slot, unsigned int device_id,
                            unsigned int iteration, unsigned int root_graph_id, const bool &is_output,
                            std::vector<std::shared_ptr<TensorData>> *result_list, bool *no_mem_to_read);

  void ReadDumpedTensorAsync(const std::string &specific_dump_dir, const std::string &prefix_dump_to_check,
                             const std::string &slot_string_to_check, const std::string &backend_name, size_t slot,
                             unsigned int device_id, unsigned int iteration, unsigned int root_graph_id,
                             const bool &is_output, const std::vector<std::string> &async_file_pool,
                             std::vector<std::shared_ptr<TensorData>> *result_list, bool *no_mem_to_read);

  std::vector<std::shared_ptr<TensorData>> ReadNeededDumpedTensors(unsigned int iteration,
                                                                   std::vector<std::string> *const async_file_pool);

  const void *GetPrevTensor(const std::shared_ptr<TensorData> &tensor, bool previous_iter_tensor_needed,
                            uint32_t *prev_num_elements);

  void ReadTensorFromNpy(const std::string &tensor_name, const std::string &file_name, std::string *const tensor_type,
                         std::size_t *const size, std::vector<int64_t> *const shape,
                         std::vector<char> **const data_buffer, bool *no_mem_to_read);

  void ConvertToHostFormat(const std::map<std::string, std::vector<std::string>> &dir_to_files_map,
                           std::vector<std::string> *const result_list);

  void ProcessConvertToHostFormat(const std::vector<std::string> &files_after_convert_in_dir,
                                  const std::string &dump_key, std::vector<std::string> *const result_list,
                                  const std::string &file_format);

  void ConvertReadTensors(std::vector<std::string> backend_name, std::vector<size_t> slot,
                          std::vector<unsigned int> device_id, std::vector<unsigned int> iteration,
                          std::vector<unsigned int> root_graph_id, std::vector<std::string> *const result_list);

  void ConvertWatchPointNodes(const std::vector<std::tuple<std::string, std::string>> &proto_dump,
                              const std::string &specific_dump_dir, std::vector<std::string> *const result_list);

  void ProcessConvertList(const std::string &prefix_dump_file_name, const std::string &file_format,
                          const std::string &specific_dump_dir,
                          std::map<std::string, std::vector<std::string>> *dir_to_files_map,
                          std::vector<std::string> *const result_list);

  void GetTensorDataInfoAsync(const std::vector<std::tuple<std::string, std::string>> &proto_dump,
                              const std::string &specific_dump_dir, uint32_t iteration, uint32_t device_id,
                              uint32_t root_graph_id, const std::vector<std::string> &async_file_pool,
                              std::vector<std::shared_ptr<TensorData>> *const tensor_list);

  std::string GetStrippedFilename(const std::string &file_name);

  std::string IterationString(unsigned int iteration);
#endif
  void ReadNodesTensors(const std::vector<std::string> &name, std::vector<std::string> *ret_name,
                        std::vector<const char *> *data_ptr, std::vector<ssize_t> *data_size,
                        std::vector<unsigned int> *dtype, std::vector<std::vector<int64_t>> *const shape);

  void SearchNodesTensors(const std::vector<std::string> &name,
                          std::vector<std::tuple<std::string, std::shared_ptr<TensorData>>> *result_list);
#ifdef ONLINE_DBG_MODE
  bool IsWatchPoint(const std::string &kernel_name, const CNodePtr &kernel = nullptr) const;

  bool IsWatchPointNodeInput(const std::string &w_name, const CNodePtr &kernel) const;
#endif

  std::vector<std::shared_ptr<TensorData>> GetTensor() const;

  void AddAnalyzedTensorToCache(const bool recheck, const unsigned int id, const std::string &tensor_name);

  void EmptyCurrentTensor();

#ifdef ONLINE_DBG_MODE
  bool DumpTensorToFile(const std::string &tensor_name, bool trans_flag, const std::string &filepath,
                        const std::string &host_fmt, const std::vector<int64_t> &host_shape, TypeId host_type,
                        TypeId device_type, const std::string &addr_format, size_t slot) const;
#endif

  bool LoadNewTensor(const std::shared_ptr<TensorData> &tensor, bool keep_prev);

  void ResetLoadedTensors();
#ifdef ONLINE_DBG_MODE
  std::vector<std::shared_ptr<TensorData>> GetNodeTensor(const CNodePtr &kernel);
#endif

  // Find if any operation overflow happened on a particular node name
  bool CheckOpOverflow(std::string node_name_to_find, unsigned int device_id = 0, unsigned int root_graph_id = 0,
                       unsigned int iteration = 0);

  bool GetAttrsFromAsyncFilename(const std::string &file_name, std::string *const node_name, uint64_t *task_id,
                                 uint64_t *stream_id);

  std::string RealPath(const std::string &input_path);

  uint64_t BytestoUInt64(const std::vector<char> &buffer);

  bool TensorExistsInCurrent(const std::string &tensor_name);

  void MoveTensorCurrentToPrev(const std::string &tensor_name);

  void AppendToCacheEvictQueue(const std::string &tensor_name);

  void SetNetName(std::string net_name);

  std::string GetNetName();

  void SetDumpDir(std::string dump_dir);

  std::string GetDumpDir();

  void SetSyncMode(bool is_sync_mode);

  bool GetSyncMode();

  void SetMemLimit(uint64_t max_mem_size);

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
  bool is_sync_mode_{false};

  std::shared_ptr<TensorLoader> tensor_loader_;
};
#ifdef ONLINE_DBG_MODE
}  // namespace mindspore
#endif

#endif  // MINDSPORE_CCSRC_DEBUG_DEBUG_SERVICES_H_
