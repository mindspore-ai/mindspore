/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "debug/debug_services.h"
#include <dirent.h>
#include <algorithm>
#include <functional>
#include <fstream>
#include <future>
#include <thread>
#include <iterator>
#include <map>
#include <numeric>
#include <limits>
#include <unordered_set>
#include <utility>
#include <regex>
#include <iomanip>
#include "pybind11/stl.h"
#ifdef ONLINE_DBG_MODE
#include "include/common/debug/common.h"
#include "debug/debugger/debugger.h"
#include "include/common/debug/anf_dump_utils.h"
#include "include/common/utils/anfalgo.h"
#endif
#include "debug/utils.h"
#include "nlohmann/json.hpp"
#include "debug/debugger/tensor_summary.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace {
static constexpr const char constant_prefix[] = "Default--data-";
static constexpr const char kNpyExt[] = ".npy";
constexpr float ms_to_s = 1000.0;
constexpr int precision = 2;
static constexpr int32_t wp_progress_period = 300;
#ifdef __APPLE__
constexpr int kStrErrorNone = 0;
#else
constexpr char *kStrErrorNone = nullptr;
#endif
}  // namespace

bool IsRegFile(const std::string &file_path) {
  struct stat st;
  int ret = stat(file_path.c_str(), &st);
  if (ret != 0) {
    MS_LOG(ERROR) << "stat error for " << file_path << ", ret is: " << ret;
    return false;
  }
  return S_ISREG(st.st_mode);
}

DebugServices::DebugServices() { tensor_loader_ = std::make_shared<TensorLoader>(); }

DebugServices::DebugServices(const DebugServices &other) {
  wp_id_cache_ = other.wp_id_cache_;
  net_name_ = other.net_name_;
  dump_dir_ = other.dump_dir_;
  is_sync_mode_ = other.is_sync_mode_;
  tensor_loader_ = other.tensor_loader_;
  watchpoint_table_ = other.watchpoint_table_;
}

DebugServices &DebugServices::operator=(const DebugServices &other) {
  if (this != &other) {
    tensor_loader_ = other.tensor_loader_;
    watchpoint_table_ = other.watchpoint_table_;
  }
  return *this;
}

/*
 * Feature group: Online debugger, Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Create a watchpoint_t object and set the watchpoint's variables and add the watchpoint to the
 * watchpoint_table.
 */
void DebugServices::AddWatchpoint(
  int id, int watch_condition, float parameter, const std::vector<std::tuple<std::string, bool>> &check_node_list,
  const std::vector<parameter_t> &parameter_list,
  const std::vector<std::tuple<std::string, std::vector<uint32_t>>> *check_node_device_list,
  const std::vector<std::tuple<std::string, std::vector<uint32_t>>> *check_node_graph_list) {
  std::lock_guard<std::mutex> lg(lock_);

  watchpoint_t watchpoint_item;
  if (id < 0) {
    MS_LOG(EXCEPTION) << "The watchpoint id should be an integer greater then 0, but got " << id;
  }
  watchpoint_item.id = static_cast<unsigned int>(id);
  watchpoint_item.condition.type = static_cast<CONDITION_TYPE>(watch_condition);
  watchpoint_item.condition.parameter = parameter;
  watchpoint_item.check_node_list = check_node_list;
  // For offline debugger check_node_device_list is not nullptr.
  if (check_node_device_list != nullptr) {
    watchpoint_item.check_node_device_list = *check_node_device_list;
  }
  // For offline debugger check_node_graph_list is not nullptr.
  if (check_node_graph_list != nullptr) {
    watchpoint_item.check_node_graph_list = *check_node_graph_list;
  }
  watchpoint_item.parameter_list = parameter_list;
  watchpoint_table_[id] = watchpoint_item;
}

void DebugServices::RemoveWatchpoint(unsigned int id) {
  std::lock_guard<std::mutex> lg(lock_);
  (void)watchpoint_table_.erase(id);
}

/*
 * Feature group: Online debugger, Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Returns a tensor summary unique pointer based on the given tensor_dtype, returns nullptr if the type is
 * not supported.
 */
std::unique_ptr<ITensorSummary> GetSummaryPtr(const std::shared_ptr<TensorData> &tensor,
                                              const void *const previous_tensor_ptr, uint64_t num_elements,
                                              uint64_t prev_num_elements, int tensor_dtype) {
  MS_EXCEPTION_IF_NULL(tensor);
  switch (tensor_dtype) {
    case DbgDataType::DT_UINT8: {
      return std::make_unique<TensorSummary<uint8_t>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                      prev_num_elements);
    }
    case DbgDataType::DT_INT8: {
      return std::make_unique<TensorSummary<int8_t>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                     prev_num_elements);
    }
    case DbgDataType::DT_UINT16: {
      return std::make_unique<TensorSummary<uint16_t>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                       prev_num_elements);
    }
    case DbgDataType::DT_INT16: {
      return std::make_unique<TensorSummary<int16_t>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                      prev_num_elements);
    }
    case DbgDataType::DT_UINT32: {
      return std::make_unique<TensorSummary<uint32_t>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                       prev_num_elements);
    }
    case DbgDataType::DT_INT32:
    case DbgDataType::DT_BASE_INT: {
      return std::make_unique<TensorSummary<int32_t>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                      prev_num_elements);
    }
    case DbgDataType::DT_UINT64: {
      return std::make_unique<TensorSummary<uint64_t>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                       prev_num_elements);
    }
    case DbgDataType::DT_INT64: {
      return std::make_unique<TensorSummary<int64_t>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                      prev_num_elements);
    }
    case DbgDataType::DT_FLOAT16: {
      return std::make_unique<TensorSummary<float16>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                      prev_num_elements);
    }
    case DbgDataType::DT_FLOAT32:
    case DbgDataType::DT_BASE_FLOAT: {
      return std::make_unique<TensorSummary<float>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                    prev_num_elements);
    }
    case DbgDataType::DT_FLOAT64: {
      return std::make_unique<TensorSummary<double>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                     prev_num_elements);
    }
    case DbgDataType::DT_BOOL: {
      return std::make_unique<TensorSummary<bool>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements,
                                                   prev_num_elements);
    }
    default:
      MS_LOG(INFO) << "Unsupported tensor type";
      // return a null pointer
      return std::unique_ptr<TensorSummary<int32_t>>{};
  }
}

/*
 * Feature group: Online debugger, Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Returns TensorStat for the given tensor based on the base_summary_ptr.
 */
DebugServices::TensorStat DebugServices::GetTensorStatistics(const std::shared_ptr<TensorData> &tensor) {
  if (tensor == nullptr) {
    MS_LOG(WARNING) << "Tensor is nullptr, returning empty tensor statistics.";
    TensorStat empty_tensor_stat_data;
    return empty_tensor_stat_data;
  }
  std::unique_ptr<ITensorSummary> base_summary_ptr;
  void *previous_tensor_ptr = nullptr;
  base_summary_ptr = GetSummaryPtr(tensor, previous_tensor_ptr, tensor->GetNumElements(), 0, tensor->GetType());
  if (base_summary_ptr == nullptr) {
    MS_LOG(WARNING) << "base_summary_ptr is nullptr, returning empty tensor statistics.";
    TensorStat empty_tensor_stat_data;
    return empty_tensor_stat_data;
  }
  base_summary_ptr->TensorStatistics(tensor->GetType());
  TensorStat tensor_stat_data(tensor->GetByteSize(), tensor->GetType(), tensor->GetShape(), base_summary_ptr->is_bool(),
                              base_summary_ptr->max_value(), base_summary_ptr->min_value(),
                              base_summary_ptr->avg_value(), base_summary_ptr->count(),
                              base_summary_ptr->neg_zero_count(), base_summary_ptr->pos_zero_count(),
                              base_summary_ptr->nan_count(), base_summary_ptr->neg_inf_count(),
                              base_summary_ptr->pos_inf_count(), base_summary_ptr->zero_count());

  return tensor_stat_data;
}

#ifdef OFFLINE_DBG_MODE
/*
 * Feature group: Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Returns previous_tensor_ptr if graph hisotry file is found and the current iteration is not the first
 * run iteration for tensor's graph.
 */
const void *DebugServices::GetPrevTensor(const std::shared_ptr<TensorData> &tensor, bool previous_iter_tensor_needed,
                                         uint64_t *prev_num_elements, bool *history_not_found) {
  MS_EXCEPTION_IF_NULL(tensor);
  const void *previous_tensor_ptr = nullptr;
  std::shared_ptr<TensorData> tensor_prev;
  std::tuple<uint32_t, uint32_t> rank_and_graph = std::make_tuple(tensor->GetDeviceId(), tensor->GetRootGraphId());
  if (graphs_run_history_.find(rank_and_graph) == graphs_run_history_.end()) {
    *history_not_found = 1;
    MS_LOG(DEBUG) << "Graph run history is not available for graph: " << tensor->GetRootGraphId();
  } else if (previous_iter_tensor_needed && GetPrevIteration(tensor) != UINT32_MAX) {
    // when prev_tensor is not available, the prev iteration is set to UINT32_MAX
    // read data in offline mode
    NPYFilePool file_paths;
    ProcessedNPYFiles processed_npy_files;
    if (!is_sync_mode_) {
      ConvertReadTensors(std::vector<std::string>{tensor->GetName()}, std::vector<size_t>{tensor->GetSlot()},
                         std::vector<unsigned int>{tensor->GetDeviceId()},
                         std::vector<unsigned int>{tensor->GetPrevIteration()},
                         std::vector<unsigned int>{tensor->GetRootGraphId()}, &file_paths);
      processed_npy_files = ProcessNPYFilePool(file_paths);
    }
    std::vector<std::shared_ptr<TensorData>> result_list_prev;
    ReadDumpedTensor(std::vector<std::string>{tensor->GetName()}, std::vector<size_t>{tensor->GetSlot()},
                     std::vector<unsigned int>{tensor->GetDeviceId()},
                     std::vector<unsigned int>{tensor->GetPrevIteration()},
                     std::vector<unsigned int>{tensor->GetRootGraphId()}, std::vector<bool>{tensor->GetIsOutput()},
                     &processed_npy_files, &result_list_prev, false);
    tensor_prev = result_list_prev[0];
    if (tensor_prev->GetByteSize() == 0) {
      tensor_prev.reset();
    } else {
      previous_tensor_ptr = tensor_prev->GetDataPtr();
      *prev_num_elements = tensor_prev->GetNumElements();
    }
  }
  return previous_tensor_ptr;
}
#endif

/*
 * Feature group: Offline debugger, Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Goes through all the watchpoints in the watchpoint table. If the current tensor is in the list of
 * check_nodes, that watchpoint is added to the vector of watchpoint_to_check (vector of watchpoints that should be
 * checked for the current tensor) .
 */
void DebugServices::AddWatchPointsToCheck(bool init_dbg_suspend, bool step_end, bool recheck,
                                          const std::shared_ptr<TensorData> &tensor, bool *previous_iter_tensor_needed,
                                          std::string *const qualified_tensor_name,
                                          std::vector<watchpoint_t> *const watchpoints_to_check) {
  if (tensor == nullptr) {
    MS_LOG(DEBUG) << "tensor is nullptr.";
    return;
  }
  const auto tensor_name = tensor->GetName();
  const auto tensor_name_no_slot = tensor_name.substr(0, tensor_name.find_first_of(':'));
  const auto tensor_device_id = tensor->GetDeviceId();
  const auto tensor_root_graph_id = tensor->GetRootGraphId();
  for (auto w_table_item : watchpoint_table_) {
    auto wp = std::get<1>(w_table_item);
    // check ONLY init conditions on initial suspended state.
    // skip other conditions on initial suspended state
    if (init_dbg_suspend && (wp.condition.type != INIT)) {
      continue;
    }
    // skip init condition if not init suspend
    if ((wp.condition.type == INIT) && !init_dbg_suspend) {
      continue;
    }
    // check change conditions only on step end.
    if (wp.change_condition() && !step_end) {
      continue;
    }
    // if recheck, ignore the cache results and reanalyze everything.
    // if not a recheck, check only unanalyzed tensors
    if (!recheck) {
      std::lock_guard<std::mutex> lg(wp_lock_);
      bool wp_cache_hit = wp_id_cache_[tensor_name].count(wp.id);
      if (wp_cache_hit) {
        continue;
      }
    }
    std::string found = wp.FindQualifiedTensorName(tensor_name_no_slot, tensor_device_id, tensor_root_graph_id);
    if (!found.empty()) {
      *qualified_tensor_name = found;
      watchpoints_to_check->push_back(w_table_item.second);
#ifdef OFFLINE_DBG_MODE
      if (wp.change_condition()) {
        *previous_iter_tensor_needed = true;
      }
#endif
    }
  }
}

void DebugServices::AddAnalyzedTensorToCache(const bool recheck, const unsigned int id,
                                             const std::string &tensor_name) {
  // add analyzed tensor to cache
  if (!recheck) {
    std::lock_guard<std::mutex> lg(wp_lock_);
    (void)wp_id_cache_[tensor_name].insert(id);
  }
}

void DebugServices::SetCheckWatchpointsResult(const int chunk_id, ChunkData *chunk_data,
                                              std::vector<unsigned int> *const device_id,
                                              std::vector<unsigned int> *const root_graph_id, const int exec_order,
                                              const std::string time_stamp, const std::string &qualified_tensor_name,
                                              const std::string &tensor_slot, const watchpoint_t &wp,
                                              const unsigned int device_id_val, const unsigned int root_graph_id_val,
                                              const std::vector<parameter_t> &parameter_list,
                                              const int32_t error_code) const {
  (void)(chunk_data->chunk_exec_orders)[chunk_id].emplace_back(exec_order);
  (void)(chunk_data->chunk_names)[chunk_id].emplace_back(qualified_tensor_name);
  (void)(chunk_data->chunk_slots)[chunk_id].emplace_back(tensor_slot);
  (void)(chunk_data->chunk_conditions)[chunk_id].emplace_back(wp.condition.type);
  (void)(chunk_data->chunk_watchpoint_id)[chunk_id].emplace_back(wp.id);
  if (device_id != nullptr) {
    (void)(chunk_data->chunk_device_id)[chunk_id].emplace_back(device_id_val);
  }
  if (root_graph_id != nullptr) {
    (void)(chunk_data->chunk_root_graph_id)[chunk_id].emplace_back(root_graph_id_val);
  }
  (void)(chunk_data->chunk_parameters)[chunk_id].emplace_back(parameter_list);
  (void)(chunk_data->chunk_error_codes)[chunk_id].emplace_back(error_code);
  (void)(chunk_data->chunk_time_stamp)[chunk_id].emplace_back(time_stamp);
}

#ifdef OFFLINE_DBG_MODE
/*
 * Feature group: Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Sets and checks the OUT_OF_MEMORY error_code (for memory limit feature) and  NO_VALUE error_code (for
 * new python API feature). Sets checkwatchpoint results.
 */
void DebugServices::CheckOutofMemoryandNoValue(const bool no_mem_to_read, const bool error_on_no_value,
                                               const std::vector<watchpoint_t> watchpoints_to_check, int chunk_id,
                                               ChunkData *chunk_data, std::vector<unsigned int> *const device_id,
                                               std::vector<unsigned int> *const root_graph_id, const int exec_order,
                                               const std::string time_stamp, const std::string &qualified_tensor_name,
                                               const std::string &tensor_slot, const unsigned int device_id_val,
                                               const unsigned int root_graph_id_val,
                                               const std::vector<parameter_t> &parameter_list) const {
  bool set_is_needed = no_mem_to_read || error_on_no_value;
  int32_t error_code_to_set = 0;
  if (no_mem_to_read) {
    // bit 3 denotes failed to load tensor because tensor is oversized and no enough memory to fit in
    error_code_to_set = ITensorSummary::OUT_OF_MEMORY;
  } else if (error_on_no_value) {
    error_code_to_set = ITensorSummary::NO_VALUE;
  }
  if (set_is_needed) {
    for (auto &wp : watchpoints_to_check) {
      SetCheckWatchpointsResult(chunk_id, chunk_data, device_id, root_graph_id, exec_order, time_stamp,
                                qualified_tensor_name, tensor_slot, wp, device_id_val, root_graph_id_val,
                                parameter_list, error_code_to_set);
    }
  }
}

/*
 * Feature group: Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: After finishing checking watchpoint, set the tensor to not-in-use status (for memory control
 * feature) by pushing it to eviction candidate queue. So it can be evicted from memory anytime if the memory is
 * required by other nodes' checking. If previous_tensor exists, change their status in a pair.
 */
void DebugServices::SetTensorToNotInUse(const std::shared_ptr<TensorData> &tensor, const void *previous_tensor_ptr) {
  // set the tensor into not-in-use status in tensor_loader.
  auto tensor_name = tensor->GetName();
  std::string key_name_in_cache = tensor_name + ":" + std::to_string(tensor->GetDeviceId()) + ":" +
                                  std::to_string(tensor->GetRootGraphId()) + ":" +
                                  std::to_string(tensor->GetIsOutput()) + ":" + std::to_string(tensor->GetSlot());
  AppendToCacheEvictQueue(key_name_in_cache);
  if (previous_tensor_ptr != nullptr) {
    AppendToCacheEvictQueue(key_name_in_cache + ":prev");
  }
}
#endif

#ifdef ONLINE_DBG_MODE
/*
 * Feature group: Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Compares the current root graph id with the given graph id and returns false if they are not equal
 * for GPU mindRT and Ascend. Otherwise, it returns true. The objectives of this function are: 1) Check if tensor's
 * root_graph_id is different from current_root_graph_id and skip checkwatchpoint for the tensor if these values are
 * different. 2) Set prev_tensor_ptr to nullptr if current_root_graph_id is different from prev_root_graph_id. 3) Skip
 * reading tensor if tensor's root_graph_id is different from current_root_graph_id.
 */
bool DebugServices::CompareCurrentRootGraph(uint32_t id) const {
  auto debugger = Debugger::GetInstance();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto cur_root_graph_id = debugger->GetCurrentRootGraphId();
  if ((device_target == kGPUDevice && MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) ||
      device_target == kAscendDevice) {
    if (cur_root_graph_id != id) {
      return false;
    }
  }
  return true;
}

/*
 * Feature group: Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Returns the previous tensor pointer if the current root graph id is equal to previous root graph id and
 * prev_tensor_data is not nullptr.
 */
const void *DebugServices::PreparePrevTensor(uint64_t *prev_num_elements, const std::string &tensor_name) {
  std::shared_ptr<TensorData> prev_tensor_data;
  if (!CompareCurrentRootGraph(Debugger::GetInstance()->GetPrevRootGraphId())) {
    // not supporting watchpoints that need prev tensor for multi root graph networks.
    MS_LOG(DEBUG) << "Previous root graph is different from current root graph, setting prev_tensor to nullptr.";
    prev_tensor_data = nullptr;
  } else {
    prev_tensor_data = tensor_loader_->GetPrevTensor(tensor_name);
  }
  if (prev_tensor_data) {
    *prev_num_elements = prev_tensor_data->GetNumElements();
    return prev_tensor_data->GetDataPtr();
  }
  return nullptr;
}
#endif

void DebugServices::CheckHistoryErrorCode(int *error_code, bool history_not_found) const {
  // check history error_code only for offline debugger
  if (history_not_found) {
    *error_code = ITensorSummary::HISTORY_NOT_FOUND;  // error code for history not found
  }
}

/*
 * Feature group: Offline debugger, Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: For all the tensors in the given chunk, reads the tensors, checks all the watchpoints and sets the
 * watchpoint hit result. Checkwatchpoint process might be affected by memory limit, whether the read tensor was
 * successfully and whether we have a multi root graph scenario. All of aforementioned checks are done in this function.
 */
void DebugServices::CheckWatchpointsForTensor(ChunkData *chunk_data, ProcessedNPYFiles *const processed_npy_files,
                                              std::vector<std::shared_ptr<TensorData>> *const tensor_list, int begin,
                                              int end, int chunk_id, const bool init_dbg_suspend, const bool step_end,
                                              const bool recheck, std::vector<unsigned int> *const device_id,
                                              std::vector<unsigned int> *const root_graph_id, bool error_on_no_value) {
  int list_size = tensor_list->size();
  if (end > list_size) {
    end = list_size;
  }
  for (int i = begin; i < end; i++) {
    auto &tensor = (*tensor_list)[i];
    const auto tensor_name = tensor->GetName();
    const auto tensor_name_no_slot = tensor_name.substr(0, tensor_name.find_first_of(':'));
    const auto tensor_slot = std::to_string(tensor->GetSlot());
    std::vector<watchpoint_t> watchpoints_to_check;
    std::string qualified_tensor_name;
    bool previous_iter_tensor_needed = false;
    AddWatchPointsToCheck(init_dbg_suspend, step_end, recheck, tensor, &previous_iter_tensor_needed,
                          &qualified_tensor_name, &watchpoints_to_check);
    // no wp set on current tensor
    if (watchpoints_to_check.empty()) {
      continue;
    }
#ifdef OFFLINE_DBG_MODE
    // read data in offline mode
    bool no_mem_to_read = false;
    std::vector<std::shared_ptr<TensorData>> result_list;
    ReadDumpedTensor(std::vector<std::string>{tensor->GetName()}, std::vector<size_t>{tensor->GetSlot()},
                     std::vector<unsigned int>{tensor->GetDeviceId()},
                     std::vector<unsigned int>{tensor->GetIteration()},
                     std::vector<unsigned int>{tensor->GetRootGraphId()}, std::vector<bool>{tensor->GetIsOutput()},
                     processed_npy_files, &result_list, false, &no_mem_to_read);
    tensor = result_list[0];
    if (tensor->GetByteSize() == 0) {
      CheckOutofMemoryandNoValue(no_mem_to_read, error_on_no_value, watchpoints_to_check, chunk_id, chunk_data,
                                 device_id, root_graph_id, tensor->GetExecutionOrder(), tensor->GetTimeStamp(),
                                 qualified_tensor_name, tensor_slot, tensor->GetDeviceId(), tensor->GetRootGraphId(),
                                 std::vector<parameter_t>());
      tensor.reset();
      continue;
    }
#endif
    // no elements to analyze
    if (tensor->GetByteSize() == 0) {
      continue;
    }
    (chunk_data->chunk_tensor_byte_size)[chunk_id] += tensor->GetByteSize();
    int tensor_dtype = tensor->GetType();
    uint64_t num_elements = tensor->GetNumElements();
    uint64_t prev_num_elements = 0;
    const void *previous_tensor_ptr = nullptr;
#ifdef OFFLINE_DBG_MODE
    bool history_not_found = 0;
    previous_tensor_ptr = GetPrevTensor(tensor, previous_iter_tensor_needed, &prev_num_elements, &history_not_found);
#else
    if (!CompareCurrentRootGraph(tensor->GetRootGraphId())) {
      MS_LOG(DEBUG)
        << "Current root_graph_id is different from tensor's root_graph_id, skipping checkwatchpoints for tensor: "
        << tensor->GetName();
      continue;
    }
    previous_tensor_ptr = PreparePrevTensor(&prev_num_elements, tensor_name);
#endif
    std::unique_ptr<ITensorSummary> base_summary_ptr;
    if (!(watchpoints_to_check.size() == 1 && watchpoints_to_check[0].condition.type == IS_OVERFLOW)) {
      base_summary_ptr = GetSummaryPtr(tensor, previous_tensor_ptr, num_elements, prev_num_elements, tensor_dtype);
      if (base_summary_ptr != nullptr) {
        base_summary_ptr->SummarizeTensor(watchpoints_to_check);
      }
    }
    for (auto &wp : watchpoints_to_check) {
      bool is_hit = false;
      int error_code = 0;
      std::vector<parameter_t> parameter_list = {};
      if (wp.condition.type == IS_OVERFLOW) {
        is_hit =
          CheckOpOverflow(tensor_name_no_slot, tensor->GetDeviceId(), tensor->GetRootGraphId(), tensor->GetIteration());
      } else if (base_summary_ptr != nullptr) {
        auto item = base_summary_ptr->IsWatchpointHit(wp);
        is_hit = std::get<ITensorSummary::eHitPos>(item);
        error_code = std::get<ITensorSummary::eErrorCodePos>(item);
#ifdef OFFLINE_DBG_MODE
        CheckHistoryErrorCode(&error_code, history_not_found);
#endif
        parameter_list = std::get<ITensorSummary::eParamListPos>(item);
      }
      AddAnalyzedTensorToCache(recheck, wp.id, tensor_name);
      if (is_hit || error_code != 0) {
        SetCheckWatchpointsResult(chunk_id, chunk_data, device_id, root_graph_id, tensor->GetExecutionOrder(),
                                  tensor->GetTimeStamp(), qualified_tensor_name, tensor_slot, wp, tensor->GetDeviceId(),
                                  tensor->GetRootGraphId(), parameter_list, error_code);
      }
    }
#ifdef OFFLINE_DBG_MODE
    SetTensorToNotInUse(tensor, previous_tensor_ptr);
    // in offline mode remove the need for the data
    tensor.reset();
#endif
    (void)tensor_processed_count_.fetch_add(1, std::memory_order_relaxed);
  }
}

/*
 * Feature group: Offline debugger, Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: This function checks the watchpoints for the given tensor list by dividing the tensor list into chunks.
 * Each chunk is handled by a separate thread and then the result of check watchpoint for each thread is gathered and
 * sorted. In the end, the time for checking the watchpoint in the current step is reported.
 */
void DebugServices::CheckWatchpoints(std::vector<std::string> *const name, std::vector<std::string> *const slot,
                                     std::vector<int> *const condition, std::vector<unsigned int> *const watchpoint_id,
                                     std::vector<std::vector<parameter_t>> *const parameters,
                                     std::vector<int32_t> *const error_codes,
                                     ProcessedNPYFiles *const processed_npy_files,
                                     std::vector<std::shared_ptr<TensorData>> *const tensor_list,
                                     const bool init_dbg_suspend, const bool step_end, const bool recheck,
                                     std::vector<unsigned int> *const device_id,
                                     std::vector<unsigned int> *const root_graph_id, bool error_on_no_value) {
  std::lock_guard<std::mutex> lg(lock_);
  auto t1 = std::chrono::high_resolution_clock::now();
  if (watchpoint_table_.empty()) {
    return;
  }
  // vector to store execution order of tensors hit
  std::vector<int> exec_order;
  std::vector<std::string> time_stamps;
  size_t tensor_list_size = tensor_list->size();
  uint64_t tensor_list_byte_size = 0;
  MS_LOG(INFO) << "tensor list size: " << tensor_list_size;
  if (tensor_list_size == 0) {
    return;
  }
  if (IS_OUTPUT_ON(mindspore::kInfo)) {
    wp_progress_enabled_ = true;
    wp_progress_thread_ =
      std::make_unique<std::thread>([this, tensor_list_size]() { CheckWatchpointProgress(tensor_list_size); });
  }
  const size_t thread_num_with_mem = 16;
  const size_t thread_num_without_mem = 32;
  // default value for number of threads
  const size_t default_thread_num =
    tensor_loader_->EnableMemoryControl() ? thread_num_with_mem : thread_num_without_mem;
  size_t max_thread_num = default_thread_num;
  if (max_thread_num > tensor_list_size) {
    max_thread_num = tensor_list_size;
  }
  MS_LOG(INFO) << "Number of threads used for checkwatchpoint: " << max_thread_num;
  size_t chunk_size = tensor_list_size / max_thread_num;
  size_t remainder = tensor_list_size % max_thread_num;
  ChunkData chunk_data;
  chunk_data.chunk_exec_orders.resize(max_thread_num);
  chunk_data.chunk_names.resize(max_thread_num);
  chunk_data.chunk_slots.resize(max_thread_num);
  chunk_data.chunk_conditions.resize(max_thread_num);
  chunk_data.chunk_watchpoint_id.resize(max_thread_num);
  chunk_data.chunk_parameters.resize(max_thread_num);
  chunk_data.chunk_error_codes.resize(max_thread_num);
  chunk_data.chunk_device_id.resize(max_thread_num);
  chunk_data.chunk_root_graph_id.resize(max_thread_num);
  chunk_data.chunk_tensor_byte_size.resize(max_thread_num);
  std::fill(chunk_data.chunk_tensor_byte_size.begin(), chunk_data.chunk_tensor_byte_size.end(), 0);
  chunk_data.chunk_time_stamp.resize(max_thread_num);

  std::vector<std::future<void>> tensor_future_vec;
  size_t begin = 0;
  size_t end = begin;
  for (size_t i = 0; i < max_thread_num; i++) {
    end += chunk_size;
    if (remainder > 0) {
      end++;
      remainder--;
    }
    (void)tensor_future_vec.emplace_back(std::async(
      std::launch::async, &DebugServices::CheckWatchpointsForTensor, this, &chunk_data, processed_npy_files,
      tensor_list, begin, end, i, init_dbg_suspend, step_end, recheck, device_id, root_graph_id, error_on_no_value));
    begin = end;
  }

  SortWatchpointsInfo(&tensor_future_vec, &exec_order, &time_stamps, &tensor_list_byte_size, name, slot,

                      condition, watchpoint_id, parameters, error_codes, &chunk_data, device_id, root_graph_id);

  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> ms_double = t2 - t1;
  MS_LOG(INFO) << "tensor_list byte size is " << tensor_list_byte_size / pow(10.0, 6.0) << " MB";
  MS_LOG(INFO) << "CheckWatchpoints Took: " << std::fixed << std::setprecision(precision)
               << (ms_double.count()) / ms_to_s << "s";
  if (IS_OUTPUT_ON(mindspore::kInfo) && wp_progress_thread_ && wp_progress_thread_->joinable()) {
    wp_progress_enabled_ = false;
    wp_progress_thread_->join();
    MS_LOG(INFO) << "Join wp_progress_thread_.";
  }
}

void DebugServices::CheckWatchpointProgress(size_t tensor_list_size) {
  while (wp_progress_enabled_ && (tensor_processed_count_ != tensor_list_size)) {
    MS_LOG(INFO) << "CheckWatchpoint progress: " << tensor_processed_count_ << " tensor processed out of "
                 << tensor_list_size;
    std::this_thread::sleep_for(std::chrono::milliseconds(wp_progress_period));
  }
}

/*
 * Feature group: Offline debugger, Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Sorts the result of watchpoint hit for the online and offline debugger. This sorting for the online
 * debugger is based on the execution order and for the offline debugger is based on the time stamp.
 */
void DebugServices::SortWatchpointsInfo(std::vector<std::future<void>> *const tensor_future_vec,
                                        std::vector<int> *const exec_order, std::vector<std::string> *const time_stamps,
                                        uint64_t *const tensor_list_byte_size, std::vector<std::string> *const name,
                                        std::vector<std::string> *const slot, std::vector<int> *const condition,
                                        std::vector<unsigned int> *const watchpoint_id,
                                        std::vector<std::vector<parameter_t>> *const parameters,
                                        std::vector<int32_t> *const error_codes, ChunkData *chunk_data,
                                        std::vector<unsigned int> *const device_id,
                                        std::vector<unsigned int> *const root_graph_id) const {
  for (unsigned int i = 0; i < (*tensor_future_vec).size(); i++) {
    (*tensor_future_vec)[i].wait();
    (*tensor_future_vec)[i].get();
    for (unsigned int j = 0; j < (chunk_data->chunk_exec_orders)[i].size(); j++) {
#ifdef ONLINE_DBG_MODE
      // if the execution order is repeated,inserts the new one before the others with same execution order.
      std::vector<int>::iterator iter =
        std::lower_bound(exec_order->begin(), exec_order->end(), (chunk_data->chunk_exec_orders)[i][j]);
      int position = iter - exec_order->begin();
      (void)exec_order->emplace(iter, (chunk_data->chunk_exec_orders)[i][j]);
#endif
#ifdef OFFLINE_DBG_MODE
      std::vector<std::string>::iterator iter =
        std::lower_bound(time_stamps->begin(), time_stamps->end(), (chunk_data->chunk_time_stamp)[i][j]);
      int position = iter - time_stamps->begin();
      (void)time_stamps->emplace(iter, (chunk_data->chunk_time_stamp)[i][j]);
#endif
      (void)name->emplace(name->begin() + position, (chunk_data->chunk_names)[i][j]);
      (void)slot->emplace(slot->begin() + position, (chunk_data->chunk_slots)[i][j]);
      (void)condition->emplace(condition->begin() + position, (chunk_data->chunk_conditions)[i][j]);
      (void)watchpoint_id->emplace(watchpoint_id->begin() + position, (chunk_data->chunk_watchpoint_id)[i][j]);
      if (device_id != nullptr) {
        (void)device_id->emplace(device_id->begin() + position, (chunk_data->chunk_device_id)[i][j]);
      }
      if (root_graph_id != nullptr) {
        (void)root_graph_id->emplace(root_graph_id->begin() + position, (chunk_data->chunk_root_graph_id)[i][j]);
      }
      (void)parameters->emplace(parameters->begin() + position, (chunk_data->chunk_parameters)[i][j]);
      (void)error_codes->emplace(error_codes->begin() + position, (chunk_data->chunk_error_codes)[i][j]);
    }
    // free the memory for used vectors
    std::vector<int>().swap((chunk_data->chunk_exec_orders)[i]);
    std::vector<std::string>().swap((chunk_data->chunk_time_stamp)[i]);
    std::vector<std::string>().swap((chunk_data->chunk_names)[i]);
    std::vector<std::string>().swap((chunk_data->chunk_slots)[i]);
    std::vector<int>().swap((chunk_data->chunk_conditions)[i]);
    std::vector<unsigned int>().swap((chunk_data->chunk_watchpoint_id)[i]);
    std::vector<std::vector<parameter_t>>().swap((chunk_data->chunk_parameters)[i]);
    std::vector<int32_t>().swap((chunk_data->chunk_error_codes)[i]);
    std::vector<unsigned int>().swap((chunk_data->chunk_device_id)[i]);
    std::vector<unsigned int>().swap((chunk_data->chunk_root_graph_id)[i]);
    if ((*tensor_list_byte_size) > UINT64_MAX - (chunk_data->chunk_tensor_byte_size)[i]) {
      MS_LOG(WARNING) << (*tensor_list_byte_size) << " + " << (chunk_data->chunk_tensor_byte_size)[i]
                      << " would lead to integer overflow!";
      (*tensor_list_byte_size) = UINT64_MAX;
    } else {
      (*tensor_list_byte_size) += (chunk_data->chunk_tensor_byte_size)[i];
    }
  }
}

#ifdef OFFLINE_DBG_MODE
/*
 * Feature group: Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Read tensor info from the given file. If memory control feature is configured to be enabled, it checks
 * if the tensor can fit in memory before reading. There are two situations to return false: 1)tensor size is greater
 * than the total preset memory limit. 2) Evicting all NOT-In-USE tensors from tensor_list_map_ cannot make enough room
 * for the tensor.
 */
void DebugServices::ReadTensorFromNpy(const std::string &tensor_name, const std::string &file_name,
                                      std::string *const tensor_type, std::size_t *const size,
                                      std::vector<int64_t> *const shape, char **const data_buffer, bool *no_mem_to_read,
                                      bool is_base_request) {
  std::ifstream infile;
  std::string file_path = file_name;
  MS_LOG(INFO) << "Reading in file: " << file_path;
  infile.open(file_path.c_str(), std::ios::ate | std::ios::binary | std::ios::in);
  if (!infile.is_open()) {
    MS_LOG(ERROR) << "Failed to open file (In ReadTensorFromNpy) " << file_path << " Errno:" << errno;
    const int kMaxFilenameLength = 128;
    char err_info[kMaxFilenameLength];
    auto ret = strerror_r(errno, err_info, sizeof(err_info));
    if (ret != kStrErrorNone) {
      MS_LOG(ERROR) << " ErrInfo:" << ret;
    }
    return;
  }
  const int substr_len = 2;
  const int header_len_offset = 8;
  const int header_offset = 9;
  const int header_len_buffer_size = 2;
  const int type_offset = 10;
  // get header length
  (void)infile.seekg(0, std::ios::beg);
  auto header_len_buffer = std::make_unique<std::vector<char>>(header_len_offset + header_len_buffer_size);
  if (!infile.read(header_len_buffer->data(), header_len_offset + header_len_buffer_size)) {
    MS_LOG(ERROR) << "Failed to parse header length from " << file_path;
    return;
  }
  uint16_t header_len = *reinterpret_cast<uint16_t *>(header_len_buffer->data() + header_len_offset);
  header_len_buffer.reset();
  // read in header
  (void)infile.seekg(0, std::ios::beg);
  auto header_buffer = std::make_unique<std::vector<char>>(header_offset + header_len);
  if (!infile.read(header_buffer->data(), header_offset + header_len)) {
    MS_LOG(ERROR) << "Failed to read header from " << file_path;
    return;
  }
  std::string header(header_buffer->data() + header_offset, header_len);
  header_buffer.reset();
  std::size_t type_i = header.find("descr") + type_offset;
  if (header.length() < type_i + substr_len) {
    MS_LOG(ERROR) << "Cannot get tensor_type, header length is " << header.length();
    return;
  }
  *tensor_type = header.substr(type_i, substr_len);
  std::size_t shape_i_open = header.find("(");
  std::size_t shape_i_close = header.find(")");
  std::string shape_str = header.substr(shape_i_open + 1, shape_i_close - shape_i_open - 1);
  std::string intermediate;
  std::stringstream check_shape(shape_str);
  MS_LOG(INFO) << "Shape of " << file_name << " is: [" << shape_str << "]";
  while (getline(check_shape, intermediate, ',')) {
    int64_t shape_d = 0;
    if (!CheckStoi(&shape_d, intermediate)) {
      MS_LOG(INFO) << "Failed to get the shape from file: " << file_name << ", error in convert the string "
                   << intermediate << " into an integer.";
      return;
    }
    shape->push_back(shape_d);
  }
  std::size_t word_size = 0;
  if (!CheckStoul(&word_size, std::string(1, (*tensor_type)[1]))) {
    MS_LOG(INFO) << "Failed to get the word_size from file: " << file_name << ", error in convert the string "
                 << (*tensor_type)[1] << " into an integer.";
    return;
  }
  std::size_t data_len = std::accumulate(shape->begin(), shape->end(), 1, std::multiplies<uint64_t>());
  std::size_t data_size = data_len * word_size;
  *size = data_size;
  if (data_size == 0 || is_base_request) {
    // for base request, reading the header is enough.
    return;
  }
  // Check memory available before loading tensor into host.
  bool has_enough_memory = true;
  if (tensor_loader_->EnableMemoryControl()) {
    has_enough_memory = tensor_loader_->CheckMemoryAvailable(tensor_name, data_size);
  }
  if (!has_enough_memory) {
    MS_LOG(ERROR) << "No enough memory available for loading " << tensor_name << " into host memory.";
    *no_mem_to_read = true;
  } else {
    (void)infile.seekg(header_len + type_offset);
    *data_buffer = new char[data_size];
    if ((*data_buffer) == nullptr || !infile.read(*data_buffer, data_size)) {
      MS_LOG(ERROR) << "Unable to get tensor data from npy";
    }
  }
}

/*
 * Feature group: Offline debugger.
 * Target device group: Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: This function is to convert files in each directory from device format to host format and append the
 * converted npy file name into NPYFilePool. It's for Ascend async dump only.
 */
void DebugServices::ConvertToHostFormat(const DirMap &dir_to_files_map, NPYFilePool *const result_list) const {
  for (auto const &d : dir_to_files_map) {
    std::vector<std::string> files_to_convert_in_dir;
    std::vector<std::string> files_after_convert_in_dir;
    std::string dump_key = d.first;
    for (auto const &item : d.second) {
      std::string file_name = std::get<0>(item);
      std::string file_name_without_scope = std::get<1>(item);

      // skip the file that was converted to npy already.
      if (std::all_of(result_list->begin(), result_list->end(), [&file_name_without_scope](std::string file_found) {
            return file_found.find(file_name_without_scope) == std::string::npos;
          })) {
        // Full path for conversion.
        (void)files_to_convert_in_dir.emplace_back(dump_key + "/" + file_name);
        (void)files_after_convert_in_dir.emplace_back(file_name_without_scope);
      }
    }
    MS_LOG(INFO) << "Number of files to convert: " << files_to_convert_in_dir.size();
    if (!files_to_convert_in_dir.empty()) {
      // Look for the installation path to the convert_async package. If not found, throw exception and terminate the
      // later task.
      auto t1 = std::chrono::high_resolution_clock::now();
      {
        pybind11::gil_scoped_acquire acquire;
        try {
          auto pkg = pybind11::module::import("mindspore.offline_debug.convert_async");
          auto convert_obj = pkg.attr("AsyncDumpConverter")(pybind11::cast(files_to_convert_in_dir), dump_key);
          (void)convert_obj.attr("convert_files")();
        } catch (pybind11::error_already_set &e) {
          MS_LOG(EXCEPTION) << "Failed to convert async dump data: " << e.what();
        }
      }
      auto t2 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> ms_double = t2 - t1;
      MS_LOG(INFO) << "convert files Took: " << std::fixed << std::setprecision(precision)
                   << (ms_double.count()) / ms_to_s << "s";
      ProcessConvertToHostFormat(files_after_convert_in_dir, dump_key, result_list);
    }
  }
}

/*
 * Feature group: Offline debugger.
 * Target device group: Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: This function is to iterate through dump directory (dump_key) and search all the converted npy files and
 * append into NPYFilePool. It's for Ascend async dump only.
 */
void DebugServices::ProcessConvertToHostFormat(const std::vector<std::string> &files_after_convert_in_dir,
                                               const std::string &dump_key, NPYFilePool *const result_list) const {
  std::string real_dump_iter_dir = RealPath(dump_key);
  DIR *d_handle = opendir(real_dump_iter_dir.c_str());
  if (d_handle == nullptr) {
    MS_LOG(INFO) << "Directory " << real_dump_iter_dir << " does not exist in ConvertToHostFormat.";
    return;
  }
  struct dirent *dir = nullptr;
  while ((dir = readdir(d_handle)) != nullptr) {
    std::string name = real_dump_iter_dir + std::string("/") + std::string(dir->d_name);
    if (!IsRegFile(name)) {
      continue;
    }
    std::string candidate = dir->d_name;
    for (const std::string &file_to_find : files_after_convert_in_dir) {
      if (candidate.find(file_to_find + ".") != std::string::npos && candidate.rfind(kNpyExt) != std::string::npos) {
        // we found a converted file for this op
        std::string found_file = dump_key + "/" + candidate;
        (void)result_list->insert(found_file);
      }
    }
  }
  (void)closedir(d_handle);
}

/*
 * Feature group: Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Node name string prefixes with scope and separates with slash "/". While the npy files in the tensor
 * dump path do not include scope in their name. The objective of this function is to remove scope from the node name to
 * match the file.
 */
std::string GetNodeNameWithoutScope(const std::string &dump_style_name) {
  if (dump_style_name.empty()) {
    return "";
  }
  std::size_t last_scope_marker;
  std::string delim = "/";
  last_scope_marker = dump_style_name.rfind(delim);
  if (last_scope_marker == std::string::npos) {
    return dump_style_name;
  }
  return dump_style_name.substr(last_scope_marker + delim.size());
}

/*
 * Feature group: Offline debugger.
 * Target device group: Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: This function is to search and prepare the target npy file to be read for each node. If the found file
 * is already npy format, push it to NPYFilePool; Otherwise, use conversion tool in convert_async.py to transfer it to
 * npy format beforehand.
 */
void DebugServices::ConvertReadTensors(std::vector<std::string> backend_name, std::vector<size_t> slot,
                                       std::vector<unsigned int> device_id, std::vector<unsigned int> iteration,
                                       std::vector<unsigned int> root_graph_id, NPYFilePool *const result_list) {
  DirMap dir_to_files_map;
  for (unsigned int i = 0; i < backend_name.size(); i++) {
    // form prefix of the tensor file to read from graph pb node name
    std::string dump_style_kernel_name = backend_name[i];

    // remove slot from name
    std::size_t found_colon = dump_style_kernel_name.find_last_of(":");
    dump_style_kernel_name = dump_style_kernel_name.substr(0, found_colon);

    MS_LOG(INFO) << "Dump style kernel_name: " << dump_style_kernel_name << ", slot is: " << slot[i];
    std::string prefix_dump_file_name = GetNodeNameWithoutScope(dump_style_kernel_name);

    std::string specific_dump_dir = dump_dir_ + "/rank_" + std::to_string(device_id[i]) + "/" + net_name_ + "/" +
                                    std::to_string(root_graph_id[i]) + "/" + IterationString(iteration[i]);

    // if node name is constant, skip
    if (prefix_dump_file_name.length() > strlen(constant_prefix) &&
        prefix_dump_file_name.substr(0, strlen(constant_prefix)).compare(constant_prefix) == 0) {
      continue;
    }
    // search files in dir for the one that meets the filename prefix and read the file into memory
    std::string abspath = RealPath(specific_dump_dir);
    auto preprocess_async_result = PreProcessDumpDirAsync(abspath);
    bool is_success = std::get<0>(preprocess_async_result);
    if (!is_success) {
      // directory does not exist
      return;
    }
    ProcessConvertList(std::get<1>(preprocess_async_result), prefix_dump_file_name, specific_dump_dir,
                       &dir_to_files_map, result_list);
  }
  ConvertToHostFormat(dir_to_files_map, result_list);
}

void DebugServices::ConvertWatchPointNodes(const DumpFileMap &dump_dir_mapped_files,
                                           const std::vector<ProtoDump> &proto_dump,
                                           const std::string &specific_dump_dir, NPYFilePool *const result_list) const {
  DirMap dir_to_files_map;
  for (const auto &node : proto_dump) {
    std::string dump_name = node.dump_name;
    // search files in dir for the one that meets the filename prefix and read the file into memory
    std::string abspath = RealPath(specific_dump_dir);
    DIR *d = opendir(abspath.c_str());
    if (d == nullptr) {
      MS_LOG(INFO) << "Directory " << specific_dump_dir.c_str() << " does not exist in ConvertWatchPointNodes.";
      return;
    }
    ProcessConvertList(dump_dir_mapped_files, dump_name, specific_dump_dir, &dir_to_files_map, result_list);
    (void)closedir(d);
  }
  ConvertToHostFormat(dir_to_files_map, result_list);
}

/*
 * Feature group: Offline debugger.
 * Target device group: Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: This function is to search the dump dir and separate npy files from bin files in async dump dir.
 */
DebugServices::AsyncPreProcessResult DebugServices::PreProcessDumpDirAsync(const std::string &specific_dump_dir) const {
  // DumpFileMap for each specific dump dir (including rank, graph_id and iteration)
  DumpFileMap dump_dir_mapped_files;
  AsyncPreProcessResult async_result;
  DIR *d = opendir(specific_dump_dir.c_str());
  if (d == nullptr) {
    MS_LOG(ERROR) << "Specific dump dir does not exit for preprocessing: " << specific_dump_dir;
    std::get<0>(async_result) = false;
    std::get<1>(async_result) = dump_dir_mapped_files;
    return async_result;
  }
  struct dirent *dir = nullptr;
  while ((dir = readdir(d)) != nullptr) {
    std::string file_name = dir->d_name;
    std::string file_path = specific_dump_dir + std::string("/") + file_name;
    if (!IsRegFile(file_path)) {
      continue;
    }
    bool is_txt = file_name.rfind(".txt") != std::string::npos;
    if (is_txt) {
      // txt files in dump dir contain the list of failed converted npy files.
      MS_LOG(DEBUG) << "Skipping txt file: " << file_name;
      continue;
    }
    std::string op_name;
    bool is_npy = file_name.rfind(kNpyExt) != std::string::npos;
    auto first_dot = file_name.find('.');

    const int kSeventhFromRight = 7;
    size_t pos = file_name.rfind(".");
    for (int cnt = 1; cnt < kSeventhFromRight; cnt++) {
      pos = file_name.rfind(".", pos - 1);
    }
    size_t seventh_last_dot = pos;

    if (seventh_last_dot != std::string::npos && first_dot != std::string::npos && seventh_last_dot > first_dot) {
      // name_to_match is between first dot and seventh last dot.
      // if op_type is parameter, the op_name can have dots.
      op_name = file_name.substr(first_dot + 1, seventh_last_dot - first_dot - 1);
    }

    if (is_npy) {
      // push back the file_name with specific dump dir
      (dump_dir_mapped_files[specific_dump_dir].npy_files[op_name]).push_back(file_path);
    } else {
      // push back the file_name without specific dump dir. dump dir is the map key.
      dump_dir_mapped_files[specific_dump_dir].bin_files.push_back(file_name);
    }
  }
  (void)closedir(d);
  std::get<0>(async_result) = true;
  std::get<1>(async_result) = dump_dir_mapped_files;
  return async_result;
}

/*
 * Feature group: Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: This function is to search the dump dir for npy files.
 */
DebugServices::NPYFilePool DebugServices::PreProcessDumpDirSync(const std::string &specific_dump_dir) const {
  // npy format:
  // {dump_path}/{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}.{output_or_input_string}.{slot}.{format}.npy
  NPYFilePool npy_files;
  DIR *d = opendir(specific_dump_dir.c_str());
  if (d == nullptr) {
    MS_LOG(ERROR) << "Specific dump dir does not exit for preprocessing: " << specific_dump_dir;
    return npy_files;
  }
  struct dirent *dir = nullptr;
  while ((dir = readdir(d)) != nullptr) {
    std::string file_name = dir->d_name;
    std::string file_path = specific_dump_dir + std::string("/") + file_name;
    if (!IsRegFile(file_path)) {
      continue;
    }
    bool is_npy = file_name.rfind(kNpyExt) != std::string::npos;
    if (is_npy) {
      (void)npy_files.insert(file_path);
    }
  }
  (void)closedir(d);
  return npy_files;
}

void DebugServices::ProcessConvertList(const DumpFileMap &dump_dir_mapped_files,
                                       const std::string &prefix_dump_file_name, const std::string &specific_dump_dir,
                                       DirMap *dir_to_files_map, NPYFilePool *const result_list) const {
  MS_EXCEPTION_IF_NULL(dir_to_files_map);
  auto it = dump_dir_mapped_files.find(specific_dump_dir);
  if (it == dump_dir_mapped_files.end()) {
    // no matched file
    MS_LOG(ERROR) << "Pre-Process is not done correctly for :" << specific_dump_dir;
    return;
  }
  auto bin_files = (it->second).bin_files;
  auto npy_files = (it->second).npy_files;

  for (size_t i = 0; i < bin_files.size(); i++) {
    std::string file_name = bin_files[i];
    std::string file_name_w_o_perfix = file_name;
    auto type_pos = file_name.find('.');
    // adding dot to avoid problematic matching in the scope.
    if (type_pos == std::string::npos ||
        file_name.find(prefix_dump_file_name + ".", type_pos + 1) == std::string::npos) {
      continue;
    }
    std::size_t second_dot = file_name.find(".", file_name.find(prefix_dump_file_name + ".", type_pos + 1));
    (void)file_name_w_o_perfix.replace(type_pos + 1, second_dot - type_pos - 1, prefix_dump_file_name);
    // if file matches prefix and is in device format add to candidate files to convert.
    (*dir_to_files_map)[specific_dump_dir].push_back(std::make_tuple(file_name, file_name_w_o_perfix));
  }
  // Add the already converted npy files to result_list
  if (npy_files.find(prefix_dump_file_name) != npy_files.end()) {
    (void)std::copy(npy_files[prefix_dump_file_name].begin(), npy_files[prefix_dump_file_name].end(),
                    std::inserter(*result_list, result_list->end()));
  }
}

void DebugServices::GetTensorDataInfoAsync(const std::vector<ProtoDump> &proto_dump,
                                           const std::string &specific_dump_dir, uint32_t iteration, uint32_t device_id,
                                           uint32_t root_graph_id, const ProcessedNPYFiles &processed_async_files,
                                           std::vector<std::shared_ptr<TensorData>> *const tensor_list) {
  auto it = processed_async_files.find(specific_dump_dir);
  if (it == processed_async_files.end()) {
    MS_LOG(DEBUG) << "no npy file was found for dump directory: " << specific_dump_dir;
    return;
  }
  auto processed_files_for_dir = it->second;
  for (auto &node : proto_dump) {
    std::vector<size_t> slot_list;
    std::string dump_name = node.dump_name;
    bool output_flag = node.is_output;

    for (const auto &dump_file_attr : processed_files_for_dir) {
      if (dump_file_attr.name_to_match == dump_name && dump_file_attr.is_output == output_flag) {
        slot_list.push_back(dump_file_attr.slot);
      }
    }
    for (auto slot : slot_list) {
      // add a TensorData entry (data will be read when needed)
      std::vector<int64_t> shape;
      std::string orig_name = node.origin_node_name;
      auto tensor_data = std::make_shared<TensorData>();
      tensor_data->SetName(orig_name);
      tensor_data->SetExecutionOrder(0);
      tensor_data->SetSlot(slot);
      tensor_data->SetIteration(iteration);
      tensor_data->SetDeviceId(device_id);
      tensor_data->SetRootGraphId(root_graph_id);
      tensor_data->SetDataPtr(nullptr);
      tensor_data->SetByteSize(0);
      tensor_data->SetType("");
      tensor_data->SetShape(shape);
      tensor_data->SetIsOutput(output_flag);
      tensor_data->SetPrevIteration(GetPrevIteration(tensor_data));

      tensor_list->push_back(tensor_data);
    }
  }
}

/*
 * Feature group: Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: This function extracts the attributes like op_name and time stamp from npy file name and is used for
 * both sync and async dump.
 */
DebugServices::ProcessedNPYFiles DebugServices::ProcessNPYFilePool(const NPYFilePool &npy_file_pool) const {
  // npy file format: node_type.node_name.task_id.stream_id.timestamp.output_input.slot.format.npy
  ProcessedNPYFiles processed_files;
  if (npy_file_pool.empty()) {
    MS_LOG(WARNING) << "ProcessNPYFilePool was called for an empty NPYFilePool.";
    return processed_files;
  }
  for (const std::string &file_name : npy_file_pool) {
    std::string file_name_to_check = file_name;
    std::string specific_dump_dir;
    DumpFileAttr dump_file_attr;
    std::string output_str;
    std::string slot_str;
    auto delim = file_name.rfind("/");
    if (delim != std::string::npos) {
      specific_dump_dir = file_name.substr(0, delim);
      file_name_to_check = file_name.substr(delim + 1);
    }
    std::vector<std::tuple<size_t, size_t, std::string *>> attr_to_match;
    size_t first_dot = file_name_to_check.find(".");
    size_t last_dot = file_name_to_check.rfind(kNpyExt);
    size_t second_last_dot = file_name_to_check.rfind(".", last_dot - 1);
    size_t third_last_dot = file_name_to_check.rfind(".", second_last_dot - 1);
    size_t fourth_last_dot = file_name_to_check.rfind(".", third_last_dot - 1);
    size_t fifth_last_dot = file_name_to_check.rfind(".", fourth_last_dot - 1);
    size_t sixth_last_dot = file_name_to_check.rfind(".", fifth_last_dot - 1);
    size_t seventh_last_dot = file_name_to_check.rfind(".", sixth_last_dot - 1);
    // name_to_match is between first dot and seventh last dot.
    // if op_type is parameter, the op_name can have dots.
    auto tuple = std::make_tuple(first_dot, seventh_last_dot, &dump_file_attr.name_to_match);
    attr_to_match.push_back(tuple);
    // slot is between second and third dot from end of the file name.
    tuple = std::make_tuple(third_last_dot, second_last_dot, &slot_str);
    attr_to_match.push_back(tuple);
    // time stamp is between fourth and fifth dot from end of the file name.
    tuple = std::make_tuple(fifth_last_dot, fourth_last_dot, &dump_file_attr.time_stamp);
    attr_to_match.push_back(tuple);
    // output is between third and fourth dot from end of the file name.
    tuple = std::make_tuple(fourth_last_dot, third_last_dot, &output_str);
    attr_to_match.push_back(tuple);
    for (auto &match_item : attr_to_match) {
      CheckStringMatch(std::get<DebugServices::START_POS>(match_item), std::get<DebugServices::END_POS>(match_item),
                       std::get<DebugServices::STR_POS>(match_item), file_name_to_check);
    }

    if (!slot_str.empty() && !CheckStoull(&dump_file_attr.slot, slot_str)) {
      MS_LOG(INFO) << "Failed to get the slot from file_name: " << file_name_to_check
                   << ", error in convert the string " << slot_str << " into an integer.";
    }
    dump_file_attr.is_output = (output_str == "output");
    dump_file_attr.file_path = file_name_to_check;
    processed_files[specific_dump_dir].push_back(dump_file_attr);
  }
  return processed_files;
}

/*
 * Feature group: Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: For the two possible modes (rank and graph), this function returns the rank_id or graph_id extracted
 * from the given directory name otherwise, it returns UINT32_MAX to identify an invalid rank or graph id.
 */
uint32_t GetRankOrGraphId(const std::string &mode, const std::string &name) {
  std::regex re;
  if (mode == "rank") {
    re = "^rank_([0-9]+)$";
  } else if (mode == "graph") {
    re = "^([0-9]+)$";
  }
  std::smatch tokens;
  if (regex_match(name, tokens, re)) {
    return std::stoi(tokens[1]);
  } else {
    return UINT32_MAX;
  }
}

std::vector<uint32_t> DebugServices::GetDumpRankIdList() {
  std::vector<uint32_t> rank_id_list;
  std::string dump_dir = GetDumpDir();
  DIR *d_handle = opendir(dump_dir.c_str());
  if (d_handle == nullptr) {
    MS_LOG(ERROR) << "Dump directory does not exist.";
    return rank_id_list;
  }
  struct dirent *dir = nullptr;
  while ((dir = readdir(d_handle)) != nullptr) {
    struct stat st;
    std::string name = dump_dir + std::string("/") + std::string(dir->d_name);
    int ret = stat(name.c_str(), &st);
    if (ret != 0) {
      MS_LOG(ERROR) << "stat error, ret is: " << ret;
      (void)closedir(d_handle);
      return rank_id_list;
    }
    if (S_ISDIR(st.st_mode)) {
      std::string rank_dir_name = dir->d_name;
      uint32_t rank_id = GetRankOrGraphId("rank", rank_dir_name);
      if (rank_id != UINT32_MAX) {
        rank_id_list.push_back(rank_id);
      }
    }
  }
  (void)closedir(d_handle);
  return rank_id_list;
}

/*
 * Feature group: Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Searches the current dump directory and for each rank_id in rank_id_list extracts the existing
 * graph_ids. Then the history file is read for all the extracted graph_ids.
 */
void DebugServices::CheckDumpGraphIdList(std::vector<uint32_t> rank_id_list) {
  std::string net_name = GetNetName();
  std::string dump_dir = GetDumpDir();
  for (uint32_t rank_id : rank_id_list) {
    std::string path = dump_dir + "/rank_" + std::to_string(rank_id) + "/" + net_name;
    std::string abspath = RealPath(path);
    DIR *d_handle_rank = opendir(abspath.c_str());
    if (d_handle_rank == nullptr) {
      MS_LOG(ERROR) << "Directory for rank_id: " << rank_id << " does not exist.";
      continue;
    }
    struct dirent *direc = nullptr;
    while ((direc = readdir(d_handle_rank)) != nullptr) {
      struct stat st;
      std::string name = abspath + std::string("/") + std::string(direc->d_name);
      int ret = stat(name.c_str(), &st);
      if (ret != 0) {
        MS_LOG(ERROR) << "stat error, ret is: " << ret;
        (void)closedir(d_handle_rank);
        return;
      }
      if (S_ISDIR(st.st_mode)) {
        std::string graph_dir = direc->d_name;
        if (graph_dir == "." || graph_dir == "..") {
          continue;
        }
        uint32_t graph_id = GetRankOrGraphId("graph", graph_dir);
        if (graph_id != UINT32_MAX) {
          ReadGraphsHistory(rank_id, graph_id);
        }
      }
    }
    (void)closedir(d_handle_rank);
  }
}

void DebugServices::SetGraphsHistory() {
  // extract rank_id_list
  std::vector<uint32_t> rank_id_list = GetDumpRankIdList();
  // for each rank_id extract the graph_id list and set the dump version
  // and for each graph read the graph history file
  CheckDumpGraphIdList(rank_id_list);
}

/*
 * Feature group: Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Reads the graph history file (containing iteration numbers in which the graph was executed) and stores
 * the data in graphs_run_history_ for the given rank and graph id.
 */
void DebugServices::ReadGraphsHistory(uint32_t rank_id, uint32_t root_graph_id) {
  std::tuple<uint32_t, uint32_t> rank_and_graph(rank_id, root_graph_id);
  if (graphs_run_history_.find(rank_and_graph) != graphs_run_history_.end()) {
    // graph history was already stored for this rank_id and graph_id
    return;
  }
  std::string exec_order_path = GetDumpDir() + "/rank_" + std::to_string(rank_id) + "/execution_order/";
  std::string file_to_check = "ms_global_execution_order_graph_" + std::to_string(root_graph_id) + ".csv";
  DIR *d_handle = opendir(exec_order_path.c_str());
  if (d_handle == nullptr) {
    MS_LOG(ERROR) << "Execution order directory does not exist.";
    return;
  }
  // read file and store the info
  std::string full_path = exec_order_path + "/" + file_to_check;
  std::string checked_path = RealPath(full_path);
  if (!checked_path.empty()) {
    ReadGraphRunIter(checked_path, rank_and_graph);
  }
  (void)closedir(d_handle);
}

/*
 * Feature group: Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Returns a map with a tuple as the key (rank, graph) and a vector as the value. This vector contains a
 * tuple with two elements, the first element is the node name and the second element is whether the node is output or
 * not.
 */
std::map<std::tuple<uint32_t, uint32_t>, std::vector<std::tuple<std::string, bool>>> DebugServices::GetAllWpNodes() {
  std::map<std::tuple<uint32_t, uint32_t>, std::vector<std::tuple<std::string, bool>>> rank_and_graph_to_nodes;
  for (auto w_table_item : watchpoint_table_) {
    auto wp = std::get<1>(w_table_item);
    unsigned int index = 0;
    for (auto check_node : wp.check_node_list) {
      std::vector<uint32_t> ranks = std::get<1>(wp.check_node_device_list[index]);
      std::vector<uint32_t> graphs = std::get<1>(wp.check_node_graph_list[index]);
      // graph represents root_graph for Ascend and kernel_graph for GPU
      for (auto rank : ranks) {
        for (auto graph : graphs) {
          std::tuple<uint32_t, uint32_t> key(rank, graph);
          (rank_and_graph_to_nodes)[key].push_back(check_node);
        }
      }
      index++;
    }
  }
  return rank_and_graph_to_nodes;
}

/*
 * Feature group: Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: For the given graph and rank id, reads the graph history file, stores all the run iterations for the
 * graph in a vector and inserts it to graphs_run_history_ map.
 */
void DebugServices::ReadGraphRunIter(std::string file_path, std::tuple<uint32_t, uint32_t> rank_and_graph) {
  std::ifstream infile;
  std::string line;
  infile.open(file_path.c_str());
  if (!infile.is_open()) {
    MS_LOG(ERROR) << "Failed to open file (In ReadGraphRunIter) " << file_path << " Errno:" << errno;
    const int kMaxFilenameLength = NAME_MAX;
    char err_info[kMaxFilenameLength];
    if (strerror_r(errno, err_info, sizeof(err_info)) != kStrErrorNone) {
      MS_LOG(ERROR) << " ErrInfo:" << strerror_r(errno, err_info, sizeof(err_info));
    }

    return;
  }
  std::vector<uint32_t> run_iters_vec;
  while (std::getline(infile, line)) {
    uint32_t iter;
    std::stringstream ss(line);
    ss >> iter;
    run_iters_vec.push_back(iter);
  }
  (void)graphs_run_history_.emplace(
    std::pair<std::tuple<uint32_t, uint32_t>, std::vector<uint32_t>>(rank_and_graph, run_iters_vec));
}

/*
 * Feature group: Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Creates a tensor_data object and sets its variables based on the function arguments and add the tensor
 * to the tensor_list_map_.
 */
void DebugServices::AddToTensorData(const std::string &backend_name, const std::string &time_stamp,
                                    const std::size_t slot, const unsigned int iteration, const unsigned int device_id,
                                    const unsigned int root_graph_id, const bool is_output, const std::size_t data_size,
                                    const std::string &type_name, const std::vector<int64_t> &shape, char *buffer,
                                    std::vector<std::shared_ptr<TensorData>> *const result_list) {
  // call LoadNewTensor to store tensor in internal cache
  auto tensor_data = std::make_shared<TensorData>();
  tensor_data->SetName(backend_name);
  tensor_data->SetExecutionOrder(0);
  tensor_data->SetSlot(slot);
  tensor_data->SetIteration(iteration);
  tensor_data->SetDeviceId(device_id);
  tensor_data->SetRootGraphId(root_graph_id);
  tensor_data->SetIsOutput(is_output);
  if (buffer != nullptr) {
    tensor_data->SetDataPtr(buffer);
  } else {
    tensor_data->SetDataPtr(nullptr);
  }
  tensor_data->SetByteSize(data_size);
  tensor_data->SetType(type_name);
  tensor_data->SetShape(shape);
  tensor_data->SetTimeStamp(time_stamp);
  tensor_data->SetPrevIteration(GetPrevIteration(tensor_data));
  if (data_size > 0) {
    (void)tensor_loader_->LoadNewTensor(tensor_data, false);
  }

  // add to result_list
  result_list->push_back(tensor_data);
}

int GetNewestFileIndex(std::vector<std::string> matched_time_stamps) {
  // given the vector of matched_time_stamps, get the index of the newest time stamp.
  // this index is used to find the corresponding matched_path.
  if (matched_time_stamps.empty()) {
    return -1;
  }
  auto it = std::max_element(matched_time_stamps.begin(), matched_time_stamps.end());
  int index = it - matched_time_stamps.begin();
  return index;
}

/*
 * Feature group: Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Search files in NPYFilePool (async and async mode) for the one that meets the filename
 * prefix and read the file into memory.
 */
void DebugServices::ReadDumpedTensor(std::vector<std::string> backend_name, std::vector<size_t> slot,
                                     std::vector<unsigned int> device_id, std::vector<unsigned int> iteration,
                                     std::vector<unsigned int> root_graph_id, const std::vector<bool> &is_output,
                                     ProcessedNPYFiles *const processed_npy_files,
                                     std::vector<std::shared_ptr<TensorData>> *const result_list, bool is_base_request,
                                     bool *no_mem_to_read) {
  for (unsigned int i = 0; i < backend_name.size(); i++) {
    // form prefix of the tensor file to read from graph pb node name
    std::string dump_style_kernel_name = backend_name[i];

    // remove slot from name
    std::size_t found_colon = dump_style_kernel_name.find_last_of(":");
    dump_style_kernel_name = dump_style_kernel_name.substr(0, found_colon);

    std::string specific_dump_dir;
    bool is_cst = false;
    // prefix_dump_to_check is node name used to find corresponding dump file.
    std::string prefix_dump_to_check = GetNodeNameWithoutScope(dump_style_kernel_name);
    // if node name has prefix of "Default--data-", consider as constant, search in cst folder
    if (prefix_dump_to_check.length() > strlen(constant_prefix) &&
        prefix_dump_to_check.substr(0, strlen(constant_prefix)).compare(constant_prefix) == 0) {
      specific_dump_dir = dump_dir_ + "/rank_" + std::to_string(device_id[i]) + "/" + net_name_ + "/" +
                          std::to_string(root_graph_id[i]) + "/constants";
      is_cst = true;
      const std::string prefix = "Default--";
      prefix_dump_to_check = prefix_dump_to_check.substr(prefix.length());
    } else {
      specific_dump_dir = dump_dir_ + "/rank_" + std::to_string(device_id[i]) + "/" + net_name_ + "/" +
                          std::to_string(root_graph_id[i]) + "/" + IterationString(iteration[i]);
    }
    MS_LOG(INFO) << "specific_dump_dir " << specific_dump_dir;
    if ((is_sync_mode_ || is_cst) && processed_npy_files->find(specific_dump_dir) == processed_npy_files->end()) {
      // This case happens when ReadDumpedTensor is called from GetPrevTensor function.
      NPYFilePool npy_files = PreProcessDumpDirSync(specific_dump_dir);
      *processed_npy_files = ProcessNPYFilePool(npy_files);
    }
    ReadDumpedTensorUtils(specific_dump_dir, prefix_dump_to_check, backend_name[i], slot[i], device_id[i], iteration[i],
                          root_graph_id[i], is_output[i], *processed_npy_files, result_list, no_mem_to_read,
                          is_base_request);
  }
}
/*
 * Feature group: Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: For both sync and async dump, gets the newest matched file path and reads the npy file and add the
 * tenosr_data object to tensor_list_map_. If there is no matched file, an empty tensor_data object is created with
 * data_size = 0, empty shape and nullptr buffer.
 */
void DebugServices::ReadFileAndAddToTensor(const bool found, const std::vector<std::string> &matched_paths,
                                           const std::vector<std::string> &matched_time_stamps,
                                           const std::string &backend_name, const unsigned int device_id,
                                           const unsigned int root_graph_id, bool is_output, size_t slot,
                                           bool *no_mem_to_read, unsigned int iteration,
                                           std::vector<std::shared_ptr<TensorData>> *result_list,
                                           bool is_base_request) {
  std::string time_stamp = "";
  std::string result_path = "";
  std::string type_name = "";
  size_t data_size = 0;
  std::vector<int64_t> shape;
  char *buffer = nullptr;
  if (found) {
    int index = GetNewestFileIndex(matched_time_stamps);
    if (index >= 0) {
      result_path = matched_paths[index];
      time_stamp = matched_time_stamps[index];
    }

    std::string key_name_in_cache = backend_name + ":" + std::to_string(device_id) + ":" +
                                    std::to_string(root_graph_id) + ":" + std::to_string(is_output) + ":" +
                                    std::to_string(slot);
    ReadTensorFromNpy(key_name_in_cache, result_path, &type_name, &data_size, &shape, &buffer, no_mem_to_read,
                      is_base_request);
    AddToTensorData(backend_name, time_stamp, slot, iteration, device_id, root_graph_id, is_output, data_size,
                    type_name, shape, buffer, result_list);
  } else {
    AddToTensorData(backend_name, time_stamp, slot, iteration, device_id, root_graph_id, is_output, 0, type_name, shape,
                    buffer, result_list);
    MS_LOG(INFO) << "Target tensor has not been found.";
  }
}

/*
 * Feature group: Offline debugger.
 * Target device group: Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: Iterates through all the processed npy files for the current specific_dump_dir and looks for the files
 * that match the node_name for dump, read the newest file and add the related tensor_data object.
 */
void DebugServices::ReadDumpedTensorUtils(const std::string &specific_dump_dir, const std::string &prefix_dump_to_check,
                                          const std::string &backend_name, size_t slot, unsigned int device_id,
                                          unsigned int iteration, unsigned int root_graph_id, bool is_output,
                                          const ProcessedNPYFiles &processed_npy_files,
                                          std::vector<std::shared_ptr<TensorData>> *result_list, bool *no_mem_to_read,
                                          bool is_base_request) {
  bool found = false;
  std::vector<std::string> matched_paths;
  std::vector<std::string> matched_time_stamps;
  auto it = processed_npy_files.find(specific_dump_dir);
  // If there is no npy file found we still need to add tensor data with size 0.
  if (it == processed_npy_files.end()) {
    MS_LOG(WARNING) << "no npy files was found for dump directory: " << specific_dump_dir;
  } else {
    auto processed_files_for_dir = it->second;
    for (const auto &dump_file_attr : processed_files_for_dir) {
      std::string file_name_to_check = dump_file_attr.file_path;
      std::string full_path = specific_dump_dir + "/" + file_name_to_check;

      if (dump_file_attr.name_to_match == prefix_dump_to_check && (dump_file_attr.slot == slot) &&
          (is_output == dump_file_attr.is_output)) {
        matched_paths.push_back(full_path);
        matched_time_stamps.push_back(dump_file_attr.time_stamp);
        found = true;
      }
    }
  }
  ReadFileAndAddToTensor(found, matched_paths, matched_time_stamps, backend_name, device_id, root_graph_id, is_output,
                         slot, no_mem_to_read, iteration, result_list, is_base_request);
}

/*
 * Feature group: Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Gets a list of the nodes that should be monitored, creates a vector called proto_to_dump with nodes'
 * original names and dump style names. Then, for each node, it creates an empty tensor_data object with data_byte_size
 * = 0 and data_ptr = nullptr and add it to the tensor_list (for both sync and async dump). This tensor_list is used for
 * checkwatchpoint functions.
 */
std::vector<std::shared_ptr<TensorData>> DebugServices::ReadNeededDumpedTensors(
  unsigned int iteration, ProcessedNPYFiles *const processed_npy_files, bool error_on_no_value) {
  // get a list of nodes and the devices they are on to monitor
  std::vector<std::shared_ptr<TensorData>> tensor_list;
  std::map<std::tuple<uint32_t, uint32_t>, std::vector<std::tuple<std::string, bool>>> rank_and_graph_to_nodes =
    GetAllWpNodes();
  // scan each device/iteration dir for the watched nodes for each device, and add to tensor_list
  // as they are found
  for (auto const &rank_and_graph_item : rank_and_graph_to_nodes) {
    std::tuple<uint32_t, uint32_t> rank_and_graph = rank_and_graph_item.first;
    uint32_t rank_id = std::get<0>(rank_and_graph);
    uint32_t root_graph_id = std::get<1>(rank_and_graph);
    MS_LOG(INFO) << "Get tensor files for rank_id: " << rank_id << ", root_graph_id: " << root_graph_id;
    std::string specific_dump_dir = dump_dir_ + "/rank_" + std::to_string(rank_id) + "/" + net_name_ + "/" +
                                    std::to_string(root_graph_id) + "/" + IterationString(iteration);
    std::string real_dump_dir = RealPath(specific_dump_dir);
    if (real_dump_dir.empty()) {
      MS_LOG(INFO) << "Dump dir " << specific_dump_dir << " doesn't exist. Skit it.";
      continue;
    }
    std::vector<std::tuple<std::string, bool>> wp_nodes = rank_and_graph_item.second;
    std::vector<ProtoDump> proto_to_dump;

    // convert node names to dump style
    for (auto node : wp_nodes) {
      std::string orig_name = std::get<0>(node);
      // Remove the scope from the fully qualified name to compare for both sync and async case.
      std::string dump_style_name = GetNodeNameWithoutScope(orig_name);

      bool node_is_out = std::get<1>(node);
      ProtoDump dump_proto;
      dump_proto.origin_node_name = orig_name;
      dump_proto.dump_name = dump_style_name;
      dump_proto.is_output = node_is_out;

      if (std::find(proto_to_dump.begin(), proto_to_dump.end(), dump_proto) == proto_to_dump.end()) {
        proto_to_dump.push_back(dump_proto);
      }
    }
    if (is_sync_mode_) {
      // search files in dir for the one that meets the filename prefix and read the file into memory
      NPYFilePool npy_files = PreProcessDumpDirSync(real_dump_dir);
      auto processed_npy_files_in_rank = ProcessNPYFilePool(npy_files);
      processed_npy_files->insert(processed_npy_files_in_rank.begin(), processed_npy_files_in_rank.end());
      ProcessTensorDataSync(proto_to_dump, real_dump_dir, *processed_npy_files, iteration, rank_id, root_graph_id,
                            &tensor_list, error_on_no_value);
    } else {
      auto preprocess_async_result = PreProcessDumpDirAsync(real_dump_dir);
      // convert all files in proto_to_dump to npy and add to pool of async file names
      NPYFilePool async_file_pool;
      ConvertWatchPointNodes(std::get<1>(preprocess_async_result), proto_to_dump, real_dump_dir, &async_file_pool);
      auto processed_npy_files_in_rank = ProcessNPYFilePool(async_file_pool);
      processed_npy_files->insert(processed_npy_files_in_rank.begin(), processed_npy_files_in_rank.end());
      GetTensorDataInfoAsync(proto_to_dump, real_dump_dir, iteration, rank_id, root_graph_id, *processed_npy_files,
                             &tensor_list);
    }
  }

  return tensor_list;
}

/*
 * Feature group: Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Iterates through the dump directory and for each file it looks for a match in the file name with node
 * names in proto_to_dump vector.
 */
void DebugServices::ProcessTensorDataSync(const std::vector<ProtoDump> &proto_to_dump,
                                          const std::string &specific_dump_dir, ProcessedNPYFiles processed_npy_files,
                                          unsigned int iteration, unsigned int device_id, unsigned int root_graph_id,
                                          std::vector<std::shared_ptr<TensorData>> *const tensor_list,
                                          bool error_on_no_value) {
  ProcessedNPYFiles::const_iterator it = processed_npy_files.find(specific_dump_dir);
  if (it == processed_npy_files.end()) {
    if (error_on_no_value) {
      MS_LOG(ERROR) << "no npy files was found for dump directory: " << specific_dump_dir;
    }
    return;
  }
  auto processed_files_for_dir = it->second;
  for (const auto &dump_file_attr : processed_files_for_dir) {
    for (auto &node : proto_to_dump) {
      std::string dump_name = node.dump_name;
      if (dump_name == dump_file_attr.name_to_match && node.is_output == dump_file_attr.is_output) {
        size_t slot = dump_file_attr.slot;
        std::vector<int64_t> shape;
        std::string orig_name = node.origin_node_name;
        bool output_flag = node.is_output;

        AddToTensorData(orig_name, "", slot, iteration, device_id, root_graph_id, output_flag, 0, "", shape, nullptr,
                        tensor_list);
        break;
      }
    }
  }
}

std::string DebugServices::IterationString(unsigned int iteration) const {
  std::string iteration_string;
  bool init_dbg_suspend = (iteration == std::numeric_limits<unsigned int>::max());
  if (init_dbg_suspend) {
    iteration_string = "init";
  } else {
    iteration_string = std::to_string(iteration);
  }
  return iteration_string;
}
#endif

/*
 * Feature group: Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Searches for tensor in the loaded tensors, if the tensor is found and tensor's root_graph_id is equal to
 * current root_graph_id, it updates the given vectors.
 */
void DebugServices::ReadNodesTensors(const std::vector<std::string> &name, std::vector<std::string> *const ret_name,
                                     std::vector<const char *> *const data_ptr, std::vector<ssize_t> *const data_size,
                                     std::vector<unsigned int> *const dtype,
                                     std::vector<std::vector<int64_t>> *const shape) {
  std::vector<std::tuple<std::string, std::shared_ptr<TensorData>>> result_list;
  tensor_loader_->SearchTensors(name, &result_list);

  for (auto result : result_list) {
    if (std::get<1>(result) == nullptr) {
      continue;
    }
#ifdef ONLINE_DBG_MODE
    if (!CompareCurrentRootGraph(std::get<1>(result)->GetRootGraphId())) {
      MS_LOG(INFO) << "tensor root_graph_id: " << std::get<1>(result)->GetRootGraphId()
                   << " is different from cur_root_graph_id: " << Debugger::GetInstance()->GetCurrentRootGraphId()
                   << ".";
      MS_LOG(INFO) << "Not reading tensor: " << std::get<0>(result) << ".";
    }
#endif
    (void)ret_name->emplace_back(std::get<0>(result));
    (void)data_ptr->emplace_back(std::get<1>(result)->GetDataPtr());
    (void)data_size->emplace_back(std::get<1>(result)->GetByteSize());
    (void)dtype->emplace_back(std::get<1>(result)->GetType());
    (void)shape->emplace_back(std::get<1>(result)->GetShape());
  }
}

void DebugServices::SearchNodesTensors(const std::vector<std::string> &name,
                                       std::vector<std::tuple<std::string, std::shared_ptr<TensorData>>> *result_list) {
  if (result_list == nullptr) {
    MS_LOG(DEBUG) << "result_list is nullptr.";
    return;
  }
  tensor_loader_->SearchTensors(name, result_list);
}

#ifdef ONLINE_DBG_MODE
bool DebugServices::IsWatchPoint(const std::string &kernel_name, const CNodePtr &kernel) const {
  bool ret = false;
  for (auto w_table_item : watchpoint_table_) {
    auto check_node_list = std::get<1>(w_table_item).check_node_list;
    for (auto check_node : check_node_list) {
      std::string w_name = std::get<0>(check_node);
      bool w_type = std::get<1>(check_node);
      if ((w_type == true &&
           ((kernel_name.find(w_name) != string::npos && kernel_name.rfind(w_name, 0) == 0) || w_name == "*")) ||
          (w_type == false && (kernel_name == w_name || IsWatchPointNodeInput(w_name, kernel)))) {
        ret = true;
        return ret;
      }
    }
  }
  return ret;
}

bool DebugServices::IsWatchPointNodeInput(const std::string &w_name, const CNodePtr &kernel) const {
  if (kernel != nullptr && w_name.length() > 0) {
    auto input_size = common::AnfAlgo::GetInputTensorNum(kernel);
    for (size_t j = 0; j < input_size; ++j) {
      auto input_kernel = kernel->input(j + 1);
      std::string input_kernel_name = GetKernelNodeName(input_kernel);
      auto found = w_name.find_last_of('/');
      if (found != std::string::npos && w_name.size() > found && w_name.substr(found + 1) == input_kernel_name) {
        return true;
      }
    }
    return false;
  } else {
    return false;
  }
}
#endif

std::vector<std::shared_ptr<TensorData>> DebugServices::GetTensor() const { return tensor_loader_->GetTensor(); }

std::shared_ptr<TensorData> DebugServices::GetTensor(const std::string &tensor_name) const {
  return tensor_loader_->GetTensor(tensor_name);
}

void DebugServices::EmptyCurrentTensor() { tensor_loader_->EmptyCurrentTensor(); }

#ifdef ONLINE_DBG_MODE
bool DebugServices::DumpTensorToFile(const std::string &filepath, const std::string &tensor_name, size_t slot) const {
  return tensor_loader_->DumpTensorToFile(filepath, tensor_name, slot);
}
#endif

bool DebugServices::LoadNewTensor(const std::shared_ptr<TensorData> &tensor, bool keep_prev) {
  return tensor_loader_->LoadNewTensor(tensor, keep_prev);
}

/*
 * Feature group: Offline debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Returns the previous iteration in which tensor's graph was executed, if the current step is the first
 * run iteration for the graph or graph history file is not available it returns UINT32_MAX to identify invalid
 * prev_iteration.
 */
uint32_t DebugServices::GetPrevIteration(const std::shared_ptr<TensorData> &tensor) {
  uint32_t prev_iter;
  uint32_t rank_id = tensor->GetDeviceId();
  uint32_t root_graph_id = tensor->GetRootGraphId();
  std::tuple<uint32_t, uint32_t> rank_and_graph = std::make_tuple(rank_id, root_graph_id);
  if (graphs_run_history_.find(rank_and_graph) == graphs_run_history_.end()) {
    return UINT32_MAX;
  }
  auto it = std::find(graphs_run_history_[rank_and_graph].begin(), graphs_run_history_[rank_and_graph].end(),
                      tensor->GetIteration());
  if (it == graphs_run_history_[rank_and_graph].end()) {
    // The graph is not executed in that iteration
    return UINT32_MAX;
  } else if (it == graphs_run_history_[rank_and_graph].begin()) {
    // current iteration is the first iteration that the graph was run
    // no prev iter is available
    MS_LOG(DEBUG) << "Iteration: " << tensor->GetIteration()
                  << " is the first run iteration for tensor: " << tensor->GetName();
    return UINT32_MAX;
  }
  (void)it--;
  prev_iter = *it;
  tensor->SetPrevIteration(prev_iter);
  return prev_iter;
}

void DebugServices::ResetLoadedTensors() {
  wp_id_cache_.clear();
  MS_LOG(INFO) << "Resetting loaded tensors";
  tensor_loader_->MoveParametersCurrentToPrev();
  tensor_loader_->EmptyCurrentTensor();
  // will move parameters from previous to current map
  tensor_loader_->SwapCurrentPrev();
  overflow_ops_.clear();
}

#ifdef ONLINE_DBG_MODE
std::vector<std::shared_ptr<TensorData>> DebugServices::GetNodeTensor(const CNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  std::vector<std::shared_ptr<TensorData>> result;
  auto output_size = AnfAlgo::GetOutputTensorNum(kernel);
  auto kernel_name = GetKernelNodeName(kernel);
  for (size_t j = 0; j < output_size; ++j) {
    auto tensor_name_with_slot = kernel_name + ":" + std::to_string(j);
    auto tensor = tensor_loader_->GetTensor(tensor_name_with_slot);
    if (tensor != nullptr) {
      result.push_back(tensor);
    }
  }
  return result;
}
#endif

std::string GetOnlineOpOverflowDir() {
  // only called for online debugger mode
  // get operator overflow directory for current iteration
  std::string overflow_bin_path = "";
#ifdef ONLINE_DBG_MODE
  if (DumpJsonParser::GetInstance().path().empty()) {
    MS_LOG(INFO) << "Dump config is not set.";
    return "";
  }
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  auto cur_graph = debugger->GetGraphPtr();
  if (cur_graph == nullptr) {
    return "";
  }
  overflow_bin_path = DumpJsonParser::GetInstance().GetOpOverflowBinPath(cur_graph->root_graph_id());
  auto realpath = FileUtils::GetRealPath(overflow_bin_path.c_str());
  if (!realpath.has_value()) {
    MS_LOG(INFO) << "Get real path failed for overflow_bin_path.";
    return "";
  }
  overflow_bin_path = realpath.value() + '/';
#endif
  return overflow_bin_path;
}

void DebugServices::GetOverflowTaskStreamId(const std::string &overflow_bin_path,
                                            std::vector<std::pair<uint64_t, uint64_t>> *task_stream_hits) const {
  MS_EXCEPTION_IF_NULL(task_stream_hits);
  const std::string overflow_file_prefix = "Opdebug.Node_OpDebug.";
  MS_LOG(INFO) << "Processing debug_files path: " << overflow_bin_path;
  DIR *d = opendir(overflow_bin_path.c_str());
  if (d == nullptr) {
    MS_LOG(INFO) << "Overflow bin directory does not exist!";
  } else {
    struct dirent *dir = nullptr;
    while ((dir = readdir(d)) != nullptr) {
      std::string file_name = dir->d_name;
      if (file_name.rfind(overflow_file_prefix, 0) != 0) {
        continue;
      }
      std::string file_path = overflow_bin_path + std::string("/") + file_name;
      if (IsRegFile(file_path)) {
        // detect overflow bin file
        uint64_t task_id = 0;
        uint64_t stream_id = 0;
        if (!GetTaskIdStreamId(file_name, overflow_file_prefix, &task_id, &stream_id)) {
          continue;
        }
        MS_LOG(INFO) << "Overflow bin file" << file_name << ", task_id " << task_id << ", stream_id " << stream_id
                     << ".";
        task_stream_hits->push_back(std::make_pair(task_id, stream_id));
      }
    }
    (void)closedir(d);
  }
}

void DebugServices::GetTaskStreamIdNodeMap(
  const std::string &tensors_path, std::map<std::pair<uint64_t, uint64_t>, std::string> *task_stream_to_opnames) const {
  MS_EXCEPTION_IF_NULL(task_stream_to_opnames);
  MS_LOG(INFO) << "Processing debug_files path: " << tensors_path;
  const std::string overflow_file_prefix = "Opdebug.Node_OpDebug.";
  DIR *d = opendir(tensors_path.c_str());
  if (d == nullptr) {
    MS_LOG(INFO) << "Tensors directory does not exist!";
  } else {
    struct dirent *dir = nullptr;
    while ((dir = readdir(d)) != nullptr) {
      std::string file_name = dir->d_name;
      if (file_name.rfind(overflow_file_prefix, 0) == 0) {
        MS_LOG(INFO) << "File: " << file_name << "is not a tensor file, skip it.";
        continue;
      }
      std::string file_path = tensors_path + std::string("/") + file_name;
      if (IsRegFile(file_path)) {
        // attempt to read the file
        std::ifstream infile;
        infile.open(file_path.c_str(), std::ios::ate | std::ios::binary | std::ios::in);
        if (!infile.is_open()) {
          MS_LOG(ERROR) << "Failed to open overflow bin file " << file_name << " Errno:" << errno;
          continue;
        }
        std::string node_name;
        uint64_t task_id = 0;
        uint64_t stream_id = 0;
        // detect overflow bin file, regular bin file or npy file
        bool success_parse = GetAttrsFromFilename(file_name, &node_name, &task_id, &stream_id);
        if (success_parse) {
          task_stream_to_opnames->insert({std::make_pair(task_id, stream_id), node_name});
        }
        infile.close();
      }
    }
    (void)closedir(d);
  }
}

void DebugServices::AddOpOverflowOpNames(const std::string &overflow_bin_path, const std::string &tensors_path,
                                         std::vector<std::string> *op_names) const {
  MS_EXCEPTION_IF_NULL(op_names);
  std::map<std::pair<uint64_t, uint64_t>, std::string> task_stream_to_opname;
  std::vector<std::pair<uint64_t, uint64_t>> task_stream_hit;
  GetOverflowTaskStreamId(overflow_bin_path, &task_stream_hit);
  GetTaskStreamIdNodeMap(tensors_path, &task_stream_to_opname);

  // find the op_names with an overflow hit
  for (auto &task_stream : task_stream_hit) {
    auto op_name = task_stream_to_opname[task_stream];
    if (!op_name.empty()) {
      MS_LOG(INFO) << "Operation overflow detected in " << op_name;
      op_names->push_back(op_name);
    }
  }
}

/*
 * Feature group: Online debugger, Offline debugger.
 * Target device group: Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: Checks whether for the given node the operator overflow happened or not by checking the overflow
 * directory. This function is for async mode only.
 */
bool DebugServices::CheckOpOverflow(std::string node_name_to_find, unsigned int device_id, unsigned int root_graph_id,
                                    unsigned int iteration) {
  if (is_sync_mode_) {
    return false;
  }
  std::string overflow_bin_path = "";
  std::string tensors_path = "";
#ifdef ONLINE_DBG_MODE
  overflow_bin_path = GetOnlineOpOverflowDir();
  tensors_path = overflow_bin_path;
#else
  overflow_bin_path =
    dump_dir_ + "/rank_" + std::to_string(device_id) + "/debug_files/" + IterationString(iteration) + "/";
  overflow_bin_path = RealPath(overflow_bin_path);
  MS_LOG(INFO) << "overflow_bin_path: " << overflow_bin_path;
  tensors_path = dump_dir_ + "/rank_" + std::to_string(device_id) + "/" + net_name_ + "/" +
                 std::to_string(root_graph_id) + "/" + IterationString(iteration) + "/";
  tensors_path = RealPath(tensors_path);
  if (overflow_bin_path.empty()) {
    overflow_bin_path = tensors_path;
  }
#endif
  if (overflow_bin_path.empty() || tensors_path.empty()) {
    MS_LOG(INFO) << "Get real path failed for overflow_bin_path or tensors path.";
    return false;
  }
  // remove kernel_graph_#
  std::string op_name_find_with_path = RemoveKernelGraphPrefix(node_name_to_find);
  std::replace(op_name_find_with_path.begin(), op_name_find_with_path.end(), '/', '_');

  // remove path
  size_t last_slash = node_name_to_find.rfind("/");
  std::string op_name_find = "";
  if (last_slash != std::string::npos) {
    op_name_find = node_name_to_find.substr(last_slash + 1);
  }

  std::replace(node_name_to_find.begin(), node_name_to_find.end(), '/', '_');
  std::vector<std::string> op_names;

  std::lock_guard<std::mutex> lg(overflow_wp_lock_);
  MS_LOG(INFO) << "Searching for overflow in node " << node_name_to_find;
  auto found_overflows = overflow_ops_.find(overflow_bin_path);
  if (found_overflows != overflow_ops_.end()) {
    MS_LOG(INFO) << "Found already computed overflows for " << overflow_bin_path;
    op_names = overflow_ops_[overflow_bin_path];
  } else {
    AddOpOverflowOpNames(overflow_bin_path, tensors_path, &op_names);
    overflow_ops_[overflow_bin_path] = op_names;
  }

  // determine if overflow wp has been triggered for the op name with path (from bin file)
  if (find(op_names.begin(), op_names.end(), op_name_find_with_path) != op_names.end()) {
    MS_LOG(INFO) << "Operation overflow watchpoint triggered for  " << node_name_to_find;
    return true;
  }

  // determine if overflow wp has been triggered for the op name (from npy file)
  if (find(op_names.begin(), op_names.end(), op_name_find) != op_names.end()) {
    MS_LOG(INFO) << "Operation overflow watchpoint triggered for  " << node_name_to_find;
    return true;
  }

  return false;
}

std::string DebugServices::RemoveKernelGraphPrefix(std::string node_name_to_find) const {
  std::string op_name_to_find = node_name_to_find;
  const std::string kernel_prefix = "kernel_graph_";
  if (node_name_to_find.rfind(kernel_prefix, 0) == 0) {
    auto start_of_op_name = node_name_to_find.find("/", kernel_prefix.length());
    if (start_of_op_name != std::string::npos) {
      op_name_to_find = node_name_to_find.substr(start_of_op_name + 1);
    }
  }
  return op_name_to_find;
}

bool DebugServices::GetTaskIdStreamId(std::string file_name, std::string overflow_file_prefix, uint64_t *const task_id,
                                      uint64_t *const stream_id) const {
  size_t task_pos_start = overflow_file_prefix.length();
  size_t task_pos_end = file_name.find(".", task_pos_start);
  if (task_pos_end == std::string::npos) {
    MS_LOG(ERROR) << "Cannot extract task_id from filename: " << file_name;
    return false;
  }

  size_t stream_pos_start = task_pos_end + 1;
  size_t stream_pos_end = file_name.find(".", stream_pos_start);
  if (stream_pos_end == std::string::npos) {
    MS_LOG(ERROR) << "Cannot extract stream_id from filename: " << file_name;
    return false;
  }

  std::string task_id_str = file_name.substr(task_pos_start, task_pos_end - task_pos_start);
  std::string stream_id_str = file_name.substr(stream_pos_start, stream_pos_end - stream_pos_start);
  if (!CheckStoull(task_id, task_id_str)) {
    MS_LOG(INFO) << "Failed to get the task_id from file_name: " << file_name << ", error in convert the string "
                 << task_id_str << " into an integer.";
    return false;
  }
  if (!CheckStoull(stream_id, stream_id_str)) {
    MS_LOG(INFO) << "Failed to get the stream_id from file_name: " << file_name << ", error in convert the string "
                 << stream_id_str << " into an integer.";
    return false;
  }

  return true;
}

bool DebugServices::GetAttrsFromFilename(const std::string &file_name, std::string *const node_name,
                                         uint64_t *const task_id, uint64_t *const stream_id) const {
  // get the node_name, task_id, and stream_id from dump filename in the following two formats:
  // 1. bin file: node_type.node_name.task_id.stream_id.timestamp
  // 2. npy file: node_type.node_name.task_id.stream_id.timestamp.output_input.slot.format.npy
  // Please note that node_name might contain dot (i.e. Parameter). So to search for the location of second dot, we need
  // to search the file name from right to left.
  size_t first_dot = file_name.find(".");
  size_t fourth_dot;
  if (file_name.rfind(kNpyExt) != std::string::npos) {
    // npy format file (converted file or A+M dump file)
    size_t pos = file_name.rfind(".");
    const int kFourthFromRight = 4;
    for (int cnt = 0; cnt < kFourthFromRight; cnt++) {
      pos = file_name.rfind(".", pos - 1);
    }
    fourth_dot = pos;
  } else {
    // bin format file
    fourth_dot = file_name.rfind(".");
  }
  size_t third_dot = file_name.rfind(".", fourth_dot - 1);
  size_t second_dot = file_name.rfind(".", third_dot - 1);
  // check if dots were found
  if (first_dot == std::string::npos || second_dot == std::string::npos || third_dot == std::string::npos ||
      fourth_dot == std::string::npos) {
    return false;
  }
  // get node_name
  if (first_dot < second_dot) {
    *node_name = file_name.substr(first_dot + 1, second_dot - first_dot - 1);
  } else {
    MS_LOG(ERROR) << "filename parse error to get node_name.";
    return false;
  }
  // get task id
  if (second_dot < third_dot) {
    std::string extracted_task_id = file_name.substr(second_dot + 1, third_dot - second_dot - 1);
    if (!CheckStoull(task_id, extracted_task_id)) {
      MS_LOG(INFO) << "Failed to get the task_id from file_name: " << file_name << ", error in convert the string "
                   << extracted_task_id << " into an integer.";
      return false;
    }
  } else {
    MS_LOG(ERROR) << "Filename <" << file_name << "> parse error to get task_id.";
    return false;
  }
  // get stream id
  if (third_dot < fourth_dot) {
    std::string extracted_stream_id = file_name.substr(third_dot + 1, fourth_dot - third_dot - 1);
    if (!CheckStoull(stream_id, extracted_stream_id)) {
      MS_LOG(INFO) << "Failed to get the stream_id from file_name: " << file_name << ", error in convert the string "
                   << extracted_stream_id << " into an integer.";
      return false;
    }
  } else {
    MS_LOG(ERROR) << "Filename <" << file_name << "> parse error to get stream_id.";
    return false;
  }

  return true;
}

std::string DebugServices::RealPath(const std::string &input_path) const {
  if (input_path.length() >= PATH_MAX) {
    MS_LOG(EXCEPTION) << "The length of path: " << input_path << " exceeds limit: " << PATH_MAX;
  }

  size_t path_split_pos = input_path.find_last_of('/');

  // get real path
  char real_path[PATH_MAX] = {0};

  // input_path is dir + file_name
  if (path_split_pos != std::string::npos) {
    std::string prefix_path = input_path.substr(0, path_split_pos);
    std::string file_name = input_path.substr(path_split_pos);

    if (file_name.length() > NAME_MAX) {
      MS_LOG(EXCEPTION) << "The length of file name : " << file_name.length() << " exceeds limit: " << NAME_MAX;
    }
    if (realpath(prefix_path.c_str(), real_path) == nullptr) {
      MS_LOG(INFO) << "The dir " << prefix_path << " does not exist.";
      return "";
    }

    return std::string(real_path) + file_name;
  }

  // input_path is only file_name
  if (input_path.length() > NAME_MAX) {
    MS_LOG(EXCEPTION) << "The length of file name : " << input_path.length() << " exceeds limit: " << NAME_MAX;
  }
  if (realpath(input_path.c_str(), real_path) == nullptr) {
    MS_LOG(INFO) << "The file " << input_path << " does not exist, it will be created.";
  }

  return std::string(real_path);
}

bool DebugServices::TensorExistsInCurrent(const std::string &tensor_name) {
  return tensor_loader_->TensorExistsInCurrent(tensor_name);
}
void DebugServices::MoveTensorCurrentToPrev(const std::string &tensor_name) {
  tensor_loader_->MoveTensorCurrentToPrev(tensor_name);
}

void DebugServices::AppendToCacheEvictQueue(const std::string &tensor_name) {
  if (tensor_loader_->EnableMemoryControl()) {
    tensor_loader_->AppendToCacheEvictQueue(tensor_name);
  }
}

void DebugServices::SetNetName(std::string net_name) { this->net_name_ = net_name; }

std::string DebugServices::GetNetName() { return net_name_; }

void DebugServices::SetDumpDir(std::string dump_dir) { this->dump_dir_ = dump_dir; }

std::string DebugServices::GetDumpDir() { return dump_dir_; }

void DebugServices::SetSyncMode(bool is_sync_mode) { this->is_sync_mode_ = is_sync_mode; }

bool DebugServices::GetSyncMode() const { return is_sync_mode_; }

void DebugServices::SetMemLimit(uint64_t max_mem_size) { tensor_loader_->SetMemTotal(max_mem_size); }

}  // namespace mindspore
