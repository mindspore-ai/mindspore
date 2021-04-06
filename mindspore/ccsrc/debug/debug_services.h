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
#ifndef MINDSPORE_CCSRC_DEBUG_DEBUG_SERVICES_H_
#define MINDSPORE_CCSRC_DEBUG_DEBUG_SERVICES_H_

#ifndef OFFLINE_DBG_MODE
#define ONLINE_DBG_MODE
#endif

#ifdef OFFLINE_DBG_MODE
#include "Eigen/Core"
#include "Eigen/src/Core/arch/CUDA/Half.h"
using float16 = Eigen::half;
#include "debugger/offline_debug/offline_logger.h"
#endif

#include <math.h>
#include <vector>
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

  ~DebugServices();

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

  typedef struct condition {
    CONDITION_TYPE type;
    float parameter = 0;
  } condition_t;

  typedef struct parameter {
    std::string name;
    bool disabled;
    double_t value;
    bool hit;
    double_t actual_value;
    void Evaluate(double_t actualValue, std::string inequality_type) {
      if (std::isnan(actualValue)) return;

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
  } parameter_t;

  typedef struct watchpoint {
    unsigned int id;
    condition_t condition;
    std::vector<std::tuple<std::string, bool>> check_node_list;
    std::vector<std::tuple<std::string, std::vector<uint32_t>>> check_node_device_list;
    std::vector<std::tuple<std::string, std::vector<uint32_t>>> check_node_graph_list;
    std::vector<parameter_t> parameter_list;
    size_t location = 0;

    std::string FindQualifiedTensorName(const std::string &tensor_name) {
      std::string node_name = tensor_name.substr(0, tensor_name.find_first_of(':'));
      for (auto check_node : check_node_list) {
        std::string w_name = std::get<0>(check_node);
        bool w_type = std::get<1>(check_node);
        auto found = w_name.find_last_of('/');
        if (found != std::string::npos && w_name.substr(found + 1) == tensor_name) return w_name;
        if ((w_type && (tensor_name.find(w_name) == location || w_name == "*")) || (!w_type && node_name == w_name)) {
          return w_name;
        }
      }
      return {};
    }

    bool is_gt_wp() {
      return condition.type == MAX_GT || condition.type == MIN_GT || condition.type == MEAN_GT ||
             condition.type == SD_GT || condition.type == MAX_MIN_GT;
    }

    bool is_lt_wp() {
      return condition.type == MAX_LT || condition.type == MIN_LT || condition.type == MEAN_LT ||
             condition.type == SD_LT || condition.type == MAX_MIN_LT;
    }

    bool min_max_enabled() {
      return condition.type == MAX_LT || condition.type == MAX_GT || condition.type == MIN_LT ||
             condition.type == MIN_GT || condition.type == MAX_MIN_LT || condition.type == MAX_MIN_GT ||
             (condition.type == INIT && (!parameter_list[1].disabled || !parameter_list[2].disabled)) ||
             (condition.type == TOO_LARGE && (!parameter_list[1].disabled || !parameter_list[2].disabled)) ||
             (condition.type == TOO_SMALL && (!parameter_list[1].disabled || !parameter_list[2].disabled));
    }
    // inf or nan related condition set
    bool inf_nan_enabled() {
      return condition.type == HAS_INF || condition.type == HAS_NAN || condition.type == GENERAL_OVERFLOW;
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
    bool zero_percentage_enabled() { return condition.type == ALL_ZERO || condition.type == INIT; }

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
  } watchpoint_t;

  void AddWatchpoint(
    unsigned int id, unsigned int watch_condition, float parameter,
    const std::vector<std::tuple<std::string, bool>> &check_node_list, const std::vector<parameter_t> &parameter_list,
    const std::vector<std::tuple<std::string, std::vector<uint32_t>>> *check_node_device_list = nullptr,
    const std::vector<std::tuple<std::string, std::vector<uint32_t>>> *check_node_graph_list = nullptr);

  void RemoveWatchpoint(unsigned int id);

  void CheckWatchpoints(std::vector<std::string> *name, std::vector<std::string> *slot, std::vector<int> *condition,
                        std::vector<unsigned int> *watchpoint_id, std::vector<std::vector<parameter_t>> *parameters,
                        std::vector<int32_t> *error_code, const std::vector<std::string> &op_overflows,
                        std::vector<std::shared_ptr<TensorData>> *tensor_list, bool init_dbg_suspend,
                        const bool step_end, const bool recheck, std::vector<unsigned int> *device_id = nullptr,
                        std::vector<unsigned int> *root_graph_id = nullptr);

  void AddWatchPointsToCheck(bool init_dbg_suspend, bool step_end, bool recheck, const std::string &tensor_name,
                             const std::string &tensor_name_no_slot, bool *previous_iter_tensor_needed,
                             std::string *qualified_tensor_name, std::vector<watchpoint_t> *watchpoints_to_check);

#ifdef OFFLINE_DBG_MODE
  void GetSlotInfo(const std::string &file_name, const std::string &dump_name, const std::string &specific_dump_dir,
                   std::vector<size_t> *slot_list);

  std::size_t GetShapeTypeInfo(const std::string &specific_dump_dir, std::size_t slot,
                               const std::string &prefix_dump_file_name, std::string *file_name, std::string *type_name,
                               std::string *out_dir, std::vector<int64_t> *shape);

  void ReadDumpedTensor(std::vector<std::string> backend_name, std::vector<size_t> slot,
                        std::vector<unsigned int> device_id, std::vector<unsigned int> iteration,
                        std::vector<unsigned int> root_graph_id, std::vector<std::shared_ptr<TensorData>> *result_list);

  std::vector<std::shared_ptr<TensorData>> ReadNeededDumpedTensors(unsigned int iteration);

  void *GetPrevTensor(const std::shared_ptr<TensorData> &tensor, bool previous_iter_tensor_needed);
#endif
  void ReadNodesTensors(std::vector<std::string> name, std::vector<std::string> *ret_name,
                        std::vector<char *> *data_ptr, std::vector<ssize_t> *data_size,
                        std::vector<unsigned int> *dtype, std::vector<std::vector<int64_t>> *shape);
#ifdef ONLINE_DBG_MODE
  bool IsWatchPoint(const std::string &kernel_name, const CNodePtr &kernel = nullptr) const;

  bool IsWatchPointNodeInput(const std::string &w_name, const CNodePtr &kernel) const;
#endif
  void EmptyTensor();

  std::vector<std::shared_ptr<TensorData>> GetTensor() const;

  void AddAnalyzedTensorToCache(const bool recheck, const unsigned int id, const std::string &tensor_name);

  std::vector<std::shared_ptr<TensorData>> GetNodeTensorMap(const std::string &node_name) const;

  uint32_t GetTensorLoaderIterNum() const;

  void SetTensorLoaderIterNum(uint32_t iter_num);

  void EmptyPrevTensor();

  void EmptyCurrentTensor();

#ifdef ONLINE_DBG_MODE
  bool DumpTensorToFile(const std::string &tensor_name, bool trans_flag, const std::string &filepath,
                        const std::string &host_fmt, const std::vector<int64_t> &host_shape, TypeId host_type,
                        TypeId addr_type_id, const std::string &addr_format, size_t slot) const;
#endif

  bool LoadNewTensor(const std::shared_ptr<TensorData> &tensor, bool keep_prev);

  std::unordered_map<unsigned int, watchpoint_t> GetWatchpointTable();

  void ResetLoadedTensors();
#ifdef ONLINE_DBG_MODE
  std::vector<std::shared_ptr<TensorData>> GetNodeTensor(const CNodePtr &kernel);
#endif

  bool TensorExistsInCurrent(std::string tensor_name);

  void MoveTensorCurrentToPrev(std::string tensor_name);

  void SetNetName(std::string net_name);

  std::string GetNetName();

  void SetDumpDir(std::string dump_dir);

  std::string GetDumpDir();

  void SetSyncMode(bool is_sync_mode);

  bool GetSyncMode();

 private:
  std::mutex lock_;

  // to keep track of watchpoints that have been checked already for a tensor in current step
  std::unordered_map<std::string, std::set<int32_t>> wp_id_cache;
  std::unordered_map<unsigned int, watchpoint_t> watchpoint_table;
  std::string net_name;
  std::string dump_dir;
  bool is_sync_mode;

  TensorLoader *tensor_loader_;
};
#ifdef ONLINE_DBG_MODE
}  // namespace mindspore
#endif

#endif  // MINDSPORE_CCSRC_DEBUG_DEBUG_SERVICES_H_
