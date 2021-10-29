/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include <unordered_set>
#include <utility>
#include "pybind11/embed.h"
#include "pybind11/stl.h"
#ifdef ONLINE_DBG_MODE
#include "debug/common.h"
#include "debug/debugger/debugger.h"
#include "debug/anf_ir_utils.h"
#include "backend/session/anf_runtime_algorithm.h"
#endif
#include "debug/debugger/tensor_summary.h"
#include "utils/file_utils.h"
#ifdef ONLINE_DBG_MODE
namespace mindspore {
#endif
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

void DebugServices::AddWatchpoint(
  unsigned int id, unsigned int watch_condition, float parameter,
  const std::vector<std::tuple<std::string, bool>> &check_node_list, const std::vector<parameter_t> &parameter_list,
  const std::vector<std::tuple<std::string, std::vector<uint32_t>>> *check_node_device_list,
  const std::vector<std::tuple<std::string, std::vector<uint32_t>>> *check_node_graph_list) {
  std::lock_guard<std::mutex> lg(lock_);

  watchpoint_t watchpoint_item;
  watchpoint_item.id = id;
  watchpoint_item.condition.type = static_cast<CONDITION_TYPE>(watch_condition);
  watchpoint_item.condition.parameter = parameter;
  watchpoint_item.check_node_list = check_node_list;
  if (check_node_device_list != nullptr) {
    watchpoint_item.check_node_device_list = *check_node_device_list;
  }
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

std::unique_ptr<ITensorSummary> GetSummaryPtr(const std::shared_ptr<TensorData> &tensor,
                                              const void *const previous_tensor_ptr, uint32_t num_elements,
                                              uint32_t prev_num_elements, int tensor_dtype) {
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
const void *DebugServices::GetPrevTensor(const std::shared_ptr<TensorData> &tensor, bool previous_iter_tensor_needed,
                                         uint32_t *prev_num_elements) {
  const void *previous_tensor_ptr = nullptr;
  std::shared_ptr<TensorData> tensor_prev;
  if (previous_iter_tensor_needed && tensor->GetIteration() >= 1) {
    // read data in offline mode
    std::vector<std::string> file_paths;
    if (!is_sync_mode_) {
      ConvertReadTensors(std::vector<std::string>{tensor->GetName()}, std::vector<size_t>{tensor->GetSlot()},
                         std::vector<unsigned int>{tensor->GetDeviceId()},
                         std::vector<unsigned int>{tensor->GetIteration() - 1},
                         std::vector<unsigned int>{tensor->GetRootGraphId()}, &file_paths);
    }
    std::vector<std::shared_ptr<TensorData>> result_list_prev;
    ReadDumpedTensor(std::vector<std::string>{tensor->GetName()}, std::vector<size_t>{tensor->GetSlot()},
                     std::vector<unsigned int>{tensor->GetDeviceId()},
                     std::vector<unsigned int>{tensor->GetIteration() - 1},
                     std::vector<unsigned int>{tensor->GetRootGraphId()}, std::vector<bool>{tensor->GetIsOutput()},
                     file_paths, &result_list_prev);
    tensor_prev = result_list_prev[0];
    if (!tensor_prev->GetByteSize()) {
      tensor_prev.reset();
    } else {
      previous_tensor_ptr = tensor_prev->GetDataPtr();
      *prev_num_elements = tensor_prev->GetNumElements();
    }
  }
  return previous_tensor_ptr;
}
#endif

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
      wp_lock_.lock();
      bool wp_cache_hit = wp_id_cache_[tensor_name].count(wp.id);
      wp_lock_.unlock();
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
    wp_lock_.lock();
    (void)wp_id_cache_[tensor_name].insert(id);
    wp_lock_.unlock();
  }
}

void DebugServices::SetCheckWatchpointsResult(
  const int chunk_id, partitioned_names *const chunk_names, partitioned_names *const chunk_slots,
  partitioned_numbers *const chunk_conditions, partitioned_id *const chunk_watchpoint_id,
  partitioned_parameters *const chunk_parameters, partitioned_error_code *const chunk_error_codes,
  partitioned_numbers *const chunk_exec_orders, partitioned_names *const chunk_time_stamp,
  partitioned_id *const chunk_device_id, partitioned_id *const chunk_root_graph_id,
  std::vector<unsigned int> *const device_id, std::vector<unsigned int> *const root_graph_id, const int exec_order,
  const std::string time_stamp, const std::string &qualified_tensor_name, const std::string &tensor_slot,
  const watchpoint_t &wp, const unsigned int device_id_val, const unsigned int root_graph_id_val,
  const std::vector<parameter_t> &parameter_list, const int32_t error_code) {
  (void)(*chunk_exec_orders)[chunk_id].emplace_back(exec_order);
  (void)(*chunk_names)[chunk_id].emplace_back(qualified_tensor_name);
  (void)(*chunk_slots)[chunk_id].emplace_back(tensor_slot);
  (void)(*chunk_conditions)[chunk_id].emplace_back(wp.condition.type);
  (void)(*chunk_watchpoint_id)[chunk_id].emplace_back(wp.id);
  if (device_id != nullptr) {
    (void)(*chunk_device_id)[chunk_id].emplace_back(device_id_val);
  }
  if (root_graph_id != nullptr) {
    (void)(*chunk_root_graph_id)[chunk_id].emplace_back(root_graph_id_val);
  }
  (void)(*chunk_parameters)[chunk_id].emplace_back(parameter_list);
  (void)(*chunk_error_codes)[chunk_id].emplace_back(error_code);
  (void)(*chunk_time_stamp)[chunk_id].emplace_back(time_stamp);
}

#ifdef OFFLINE_DBG_MODE
void DebugServices::ProcessCheckpointsOutofMemory(
  const bool no_mem_to_read, const std::vector<watchpoint_t> watchpoints_to_check, int chunk_id,
  partitioned_names *const chunk_names, partitioned_names *const chunk_slots,
  partitioned_numbers *const chunk_conditions, partitioned_id *const chunk_watchpoint_id,
  partitioned_parameters *const chunk_parameters, partitioned_error_code *const chunk_error_codes,
  partitioned_numbers *const chunk_exec_orders, partitioned_names *const chunk_time_stamp,
  partitioned_id *const chunk_device_id, partitioned_id *const chunk_root_graph_id,
  std::vector<unsigned int> *const device_id, std::vector<unsigned int> *const root_graph_id, const int exec_order,
  const std::string time_stamp, const std::string &qualified_tensor_name, const std::string &tensor_slot,
  const unsigned int device_id_val, const unsigned int root_graph_id_val,
  const std::vector<parameter_t> &parameter_list) {
  if (no_mem_to_read) {
    // bit 3 denotes failed to load tensor because tensor is oversized and no enough memory to fit in
    int32_t oversize_error_code = 8;
    for (auto &wp : watchpoints_to_check) {
      SetCheckWatchpointsResult(chunk_id, chunk_names, chunk_slots, chunk_conditions, chunk_watchpoint_id,
                                chunk_parameters, chunk_error_codes, chunk_exec_orders, chunk_time_stamp,
                                chunk_device_id, chunk_root_graph_id, device_id, root_graph_id, exec_order, time_stamp,
                                qualified_tensor_name, tensor_slot, wp, device_id_val, root_graph_id_val,
                                parameter_list, oversize_error_code);
    }
  }
}
#endif

void DebugServices::CheckWatchpointsForTensor(
  partitioned_names *const chunk_names, partitioned_names *const chunk_slots,
  partitioned_numbers *const chunk_conditions, partitioned_id *const chunk_watchpoint_id,
  partitioned_parameters *const chunk_parameters, partitioned_error_code *const chunk_error_codes,
  const std::vector<std::string> &op_overflows, const std::vector<std::string> &async_file_pool,
  partitioned_numbers *const chunk_exec_orders, std::vector<std::shared_ptr<TensorData>> *const tensor_list, int begin,
  int end, int chunk_id, const bool init_dbg_suspend, const bool step_end, const bool recheck,
  partitioned_id *const chunk_device_id, partitioned_id *const chunk_root_graph_id,
  std::vector<uint64_t> *const chunk_tensor_byte_size, partitioned_names *const chunk_time_stamp,
  std::vector<unsigned int> *const device_id, std::vector<unsigned int> *const root_graph_id) {
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
                     async_file_pool, &result_list, &no_mem_to_read);
    tensor = result_list[0];
    if (!tensor->GetByteSize()) {
      ProcessCheckpointsOutofMemory(
        no_mem_to_read, watchpoints_to_check, chunk_id, chunk_names, chunk_slots, chunk_conditions, chunk_watchpoint_id,
        chunk_parameters, chunk_error_codes, chunk_exec_orders, chunk_time_stamp, chunk_device_id, chunk_root_graph_id,
        device_id, root_graph_id, tensor->GetExecutionOrder(), tensor->GetTimeStamp(), qualified_tensor_name,
        tensor_slot, tensor->GetDeviceId(), tensor->GetRootGraphId(), std::vector<parameter_t>());
      tensor.reset();
      continue;
    }
#endif
    // no elements to analyze
    if (tensor->GetByteSize() == 0) {
      continue;
    }
    (*chunk_tensor_byte_size)[chunk_id] += tensor->GetByteSize();
    int tensor_dtype = tensor->GetType();
    uint32_t num_elements = tensor->GetNumElements();
    uint32_t prev_num_elements = 0;
    const void *previous_tensor_ptr = nullptr;
#ifdef OFFLINE_DBG_MODE
    previous_tensor_ptr = GetPrevTensor(tensor, previous_iter_tensor_needed, &prev_num_elements);
#else
    std::shared_ptr<TensorData> prev_tensor_data = tensor_loader_->GetPrevTensor(tensor_name);
    if (prev_tensor_data) {
      previous_tensor_ptr = prev_tensor_data->GetDataPtr();
      prev_num_elements = prev_tensor_data->GetNumElements();
    }
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
        parameter_list = std::get<ITensorSummary::eParamListPos>(item);
      }
      AddAnalyzedTensorToCache(recheck, wp.id, tensor_name);
      if (is_hit || error_code) {
        SetCheckWatchpointsResult(
          chunk_id, chunk_names, chunk_slots, chunk_conditions, chunk_watchpoint_id, chunk_parameters,
          chunk_error_codes, chunk_exec_orders, chunk_time_stamp, chunk_device_id, chunk_root_graph_id, device_id,
          root_graph_id, tensor->GetExecutionOrder(), tensor->GetTimeStamp(), qualified_tensor_name, tensor_slot, wp,
          tensor->GetDeviceId(), tensor->GetRootGraphId(), parameter_list, error_code);
      }
    }

#ifdef OFFLINE_DBG_MODE
    // set the tensor into not-in-use status in tensor_loader.
    std::string key_name_in_cache = tensor_name + ":" + std::to_string(tensor->GetDeviceId()) + ":" +
                                    std::to_string(tensor->GetRootGraphId()) + ":" +
                                    std::to_string(tensor->GetIsOutput()) + ":" + std::to_string(tensor->GetSlot());
    AppendToCacheEvictQueue(key_name_in_cache);
    if (previous_tensor_ptr != nullptr) {
      AppendToCacheEvictQueue(key_name_in_cache + ":prev");
    }
    // in offline mode remove the need for the data
    tensor.reset();
#endif
  }
}
void DebugServices::CheckWatchpoints(
  std::vector<std::string> *const name, std::vector<std::string> *const slot, std::vector<int> *const condition,
  std::vector<unsigned int> *const watchpoint_id, std::vector<std::vector<parameter_t>> *const parameters,
  std::vector<int32_t> *const error_codes, const std::vector<std::string> &op_overflows,
  const std::vector<std::string> &async_file_pool, std::vector<std::shared_ptr<TensorData>> *const tensor_list,
  const bool init_dbg_suspend, const bool step_end, const bool recheck, std::vector<unsigned int> *const device_id,
  std::vector<unsigned int> *const root_graph_id) {
  std::lock_guard<std::mutex> lg(lock_);
  auto t1 = std::chrono::high_resolution_clock::now();
  if (watchpoint_table_.empty()) {
    return;
  }
  // vector to store execution order of tensors hit
  std::vector<int> exec_order;
  std::vector<std::string> time_stamps;
  int tensor_list_size = tensor_list->size();
  uint64_t tensor_list_byte_size = 0;
  MS_LOG(INFO) << "tensor list size: " << tensor_list_size;
  if (tensor_list_size <= 0) {
    return;
  }
  // default value for number of threads
  const int default_thread_num = 16;
  int max_thread_num = default_thread_num;
  if (max_thread_num > tensor_list_size) {
    max_thread_num = tensor_list_size;
  }
  MS_LOG(INFO) << "Number of threads used for checkwatchpoint: " << max_thread_num;
  int chunk_size = tensor_list_size / max_thread_num;
  int remainder = tensor_list_size % max_thread_num;
  partitioned_numbers chunk_exec_orders(max_thread_num);
  partitioned_names chunk_names(max_thread_num);
  partitioned_names chunk_slots(max_thread_num);
  partitioned_numbers chunk_conditions(max_thread_num);
  partitioned_id chunk_watchpoint_id(max_thread_num);
  partitioned_parameters chunk_parameters(max_thread_num);
  partitioned_error_code chunk_error_codes(max_thread_num);
  partitioned_id chunk_device_id(max_thread_num);
  partitioned_id chunk_root_graph_id(max_thread_num);
  std::vector<uint64_t> chunk_tensor_byte_size(max_thread_num, 0);
  partitioned_names chunk_time_stamp(max_thread_num);

  std::vector<std::future<void>> tensor_future_vec;
  int begin = 0;
  int end = begin;
  for (int i = 0; i < max_thread_num; i++) {
    end += chunk_size;
    if (remainder > 0) {
      end++;
      remainder--;
    }
    (void)tensor_future_vec.emplace_back(std::async(
      std::launch::async, &DebugServices::CheckWatchpointsForTensor, this, &chunk_names, &chunk_slots,
      &chunk_conditions, &chunk_watchpoint_id, &chunk_parameters, &chunk_error_codes, op_overflows, async_file_pool,
      &chunk_exec_orders, tensor_list, begin, end, i, init_dbg_suspend, step_end, recheck, &chunk_device_id,
      &chunk_root_graph_id, &chunk_tensor_byte_size, &chunk_time_stamp, device_id, root_graph_id));
    begin = end;
  }

  SortWatchpointsInfo(&tensor_future_vec, &exec_order, &time_stamps, &tensor_list_byte_size, name, slot, condition,
                      watchpoint_id, parameters, error_codes, &chunk_names, &chunk_slots, &chunk_conditions,
                      &chunk_watchpoint_id, &chunk_parameters, &chunk_error_codes, &chunk_exec_orders,
                      &chunk_time_stamp, &chunk_tensor_byte_size, &chunk_device_id, &chunk_root_graph_id, device_id,
                      root_graph_id);

  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> ms_double = t2 - t1;
  MS_LOG(INFO) << "tensor_list byte size is " << tensor_list_byte_size / pow(10.0, 6.0) << " MB";
  MS_LOG(INFO) << "CheckWatchpoints Took: " << ms_double.count() / 1000 << "s";
}

void DebugServices::SortWatchpointsInfo(
  std::vector<std::future<void>> *const tensor_future_vec, std::vector<int> *const exec_order,
  std::vector<std::string> *const time_stamps, uint64_t *const tensor_list_byte_size,
  std::vector<std::string> *const name, std::vector<std::string> *const slot, std::vector<int> *const condition,
  std::vector<unsigned int> *const watchpoint_id, std::vector<std::vector<parameter_t>> *const parameters,
  std::vector<int32_t> *const error_codes, partitioned_names *const chunk_names, partitioned_names *const chunk_slots,
  partitioned_numbers *const chunk_conditions, partitioned_id *const chunk_watchpoint_id,
  partitioned_parameters *const chunk_parameters, partitioned_error_code *const chunk_error_codes,
  partitioned_numbers *const chunk_exec_orders, partitioned_names *const chunk_time_stamp,
  std::vector<uint64_t> *const chunk_tensor_byte_size, partitioned_id *const chunk_device_id,
  partitioned_id *const chunk_root_graph_id, std::vector<unsigned int> *const device_id,
  std::vector<unsigned int> *const root_graph_id) {
  for (unsigned int i = 0; i < (*tensor_future_vec).size(); i++) {
    (*tensor_future_vec)[i].wait();
    (*tensor_future_vec)[i].get();
    for (unsigned int j = 0; j < (*chunk_exec_orders)[i].size(); j++) {
#ifdef ONLINE_DBG_MODE
      // if the execution order is repeated,inserts the new one before the others with same execution order.
      std::vector<int>::iterator iter =
        std::lower_bound(exec_order->begin(), exec_order->end(), (*chunk_exec_orders)[i][j]);
      int position = iter - exec_order->begin();
      (void)exec_order->emplace(iter, (*chunk_exec_orders)[i][j]);
#endif
#ifdef OFFLINE_DBG_MODE
      std::vector<std::string>::iterator iter =
        std::lower_bound(time_stamps->begin(), time_stamps->end(), (*chunk_time_stamp)[i][j]);
      int position = iter - time_stamps->begin();
      (void)time_stamps->emplace(iter, (*chunk_time_stamp)[i][j]);
#endif
      (void)name->emplace(name->begin() + position, (*chunk_names)[i][j]);
      (void)slot->emplace(slot->begin() + position, (*chunk_slots)[i][j]);
      (void)condition->emplace(condition->begin() + position, (*chunk_conditions)[i][j]);
      (void)watchpoint_id->emplace(watchpoint_id->begin() + position, (*chunk_watchpoint_id)[i][j]);
      if (device_id != nullptr) {
        (void)device_id->emplace(device_id->begin() + position, (*chunk_device_id)[i][j]);
      }
      if (root_graph_id != nullptr) {
        (void)root_graph_id->emplace(root_graph_id->begin() + position, (*chunk_root_graph_id)[i][j]);
      }
      (void)parameters->emplace(parameters->begin() + position, (*chunk_parameters)[i][j]);
      (void)error_codes->emplace(error_codes->begin() + position, (*chunk_error_codes)[i][j]);
    }
    // free the memory for used vectors
    std::vector<int>().swap((*chunk_exec_orders)[i]);
    std::vector<std::string>().swap((*chunk_time_stamp)[i]);
    std::vector<std::string>().swap((*chunk_names)[i]);
    std::vector<std::string>().swap((*chunk_slots)[i]);
    std::vector<int>().swap((*chunk_conditions)[i]);
    std::vector<unsigned int>().swap((*chunk_watchpoint_id)[i]);
    std::vector<std::vector<parameter_t>>().swap((*chunk_parameters)[i]);
    std::vector<int32_t>().swap((*chunk_error_codes)[i]);
    std::vector<unsigned int>().swap((*chunk_device_id)[i]);
    std::vector<unsigned int>().swap((*chunk_root_graph_id)[i]);
    (*tensor_list_byte_size) += (*chunk_tensor_byte_size)[i];
  }
}

#ifdef OFFLINE_DBG_MODE
void DebugServices::ReadTensorFromNpy(const std::string &tensor_name, const std::string &file_name,
                                      std::string *const tensor_type, std::size_t *const size,
                                      std::vector<int64_t> *const shape, std::vector<char> **const data_buffer,
                                      bool *no_mem_to_read) {
  std::ifstream infile;
  std::string file_path = file_name;
  MS_LOG(INFO) << "Reading in file: " << file_path;
  infile.open(file_path.c_str(), std::ios::ate | std::ios::binary | std::ios::in);
  if (!infile.is_open()) {
    MS_LOG(ERROR) << "Failed to open file (In ReadTensorFromNpy) " << file_path << " Errno:" << errno;
    const int kMaxFilenameLength = 128;
    char err_info[kMaxFilenameLength];
    auto ret = strerror_r(errno, err_info, sizeof(err_info));
    if (ret != nullptr) {
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
  auto header_buffer = std::make_unique<std::vector<char>>(header_len_offset + header_len);
  if (!infile.read(header_buffer->data(), header_len_offset + header_len)) {
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
    shape->push_back(std::stoi(intermediate));
  }
  std::size_t word_size = std::stoul(std::string(1, (*tensor_type)[1]));
  std::size_t data_len = std::accumulate(shape->begin(), shape->end(), 1, std::multiplies<uint64_t>());
  std::size_t data_size = data_len * word_size;
  if (!data_size) {
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
    *data_buffer = new std::vector<char>(data_size);
    if ((*data_buffer) == nullptr || !infile.read((*data_buffer)->data(), data_size)) {
      MS_LOG(ERROR) << "Unable to get tensor data from npy";
    }
    *size = data_size;
  }
}

void DebugServices::ConvertToHostFormat(const std::map<std::string, std::vector<std::string>> &dir_to_files_map,
                                        std::vector<std::string> *const result_list) {
  std::string file_format = "npy";
  for (auto const &d : dir_to_files_map) {
    std::vector<std::string> files_to_convert_in_dir;
    std::vector<std::string> files_after_convert_in_dir;
    std::string dump_key = d.first;
    for (auto const &file_name : d.second) {
      bool already_converted = false;
      // Remove scope from the file_name for matching files converted by mindinsight tool.
      std::size_t found_first_dot = file_name.find(".");
      std::size_t found_last_underscore = file_name.find_last_of("_");
      std::string file_name_without_scope = file_name;
      if (found_last_underscore != std::string::npos && found_last_underscore > found_first_dot) {
        file_name_without_scope =
          file_name_without_scope.erase(found_first_dot + 1, found_last_underscore - found_first_dot);
      }
      for (std::string &file_found : *result_list) {
        if (file_found.find(file_name_without_scope) != std::string::npos) {
          already_converted = true;
          break;
        }
      }
      if (!already_converted) {
        (void)files_to_convert_in_dir.emplace_back(dump_key + "/" + file_name);
        (void)files_after_convert_in_dir.emplace_back(dump_key + "/" + file_name_without_scope);
      }
    }
    MS_LOG(INFO) << "Number of files to convert: " << files_to_convert_in_dir.size();
    if (!files_to_convert_in_dir.empty()) {
      // Look for the installation path to the conver_async package. If not found, throw exception and terminate the
      // later task.
      try {
        auto pkg = pybind11::module::import("mindspore.offline_debug.convert_async");
        auto convert_obj = pkg.attr("AsyncDumpConverter")(pybind11::cast(files_to_convert_in_dir), dump_key);
        (void)convert_obj.attr("convert_files")();
      } catch (pybind11::error_already_set &e) {
        MS_LOG(EXCEPTION) << "Failed to convert async dump data: " << e.what();
      }
      ProcessConvertToHostFormat(files_after_convert_in_dir, dump_key, result_list, file_format);
    }
  }
}

void DebugServices::ProcessConvertToHostFormat(const std::vector<std::string> &files_after_convert_in_dir,
                                               const std::string &dump_key, std::vector<std::string> *const result_list,
                                               const std::string &file_format) {
  std::string real_dump_iter_dir = RealPath(dump_key);
  DIR *d_handle = opendir(real_dump_iter_dir.c_str());
  if (d_handle == nullptr) {
    MS_LOG(ERROR) << "Directory does not exit in ConvertToHostFormat.";
    return;
  }
  struct dirent *dir = nullptr;
  while ((dir = readdir(d_handle)) != nullptr) {
    if (dir->d_type == DT_REG) {
      std::string candidate = dir->d_name;
      for (const std::string &file_to_find : files_after_convert_in_dir) {
        std::string file_n = file_to_find;
        auto last_slash_pos = file_to_find.find_last_of("\\/");
        if (last_slash_pos != std::string::npos) {
          file_n = file_to_find.substr(last_slash_pos + 1);
        }
        if (candidate.find(file_n) != std::string::npos && candidate.rfind(file_format) != std::string::npos) {
          // we found a converted file for this op
          std::string found_file = dump_key + "/" + candidate;
          if (std::find(result_list->begin(), result_list->end(), found_file) == result_list->end()) {
            result_list->push_back(found_file);
          }
        }
      }
    }
  }
  (void)closedir(d_handle);
}

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

void DebugServices::ConvertReadTensors(std::vector<std::string> backend_name, std::vector<size_t> slot,
                                       std::vector<unsigned int> device_id, std::vector<unsigned int> iteration,
                                       std::vector<unsigned int> root_graph_id,
                                       std::vector<std::string> *const result_list) {
  std::string file_format = "npy";
  std::map<std::string, std::vector<std::string>> dir_to_files_map;
  for (unsigned int i = 0; i < backend_name.size(); i++) {
    // form prefix of the tensor file to read from graph pb node name
    std::string dump_style_kernel_name = backend_name[i];

    // remove slot from name
    std::size_t found_colon = dump_style_kernel_name.find_last_of(":");
    dump_style_kernel_name = dump_style_kernel_name.substr(0, found_colon);

    std::string prefix_dump_file_name = GetNodeNameWithoutScope(dump_style_kernel_name);

    std::string specific_dump_dir = dump_dir_ + "/rank_" + std::to_string(device_id[i]) + "/" + net_name_ + "/" +
                                    std::to_string(root_graph_id[i]) + "/" + IterationString(iteration[i]);

    // search files in dir for the one that meets the filename prefix and read the file into memory
    std::string abspath = RealPath(specific_dump_dir);
    DIR *d = opendir(abspath.c_str());
    if (d == nullptr) {
      MS_LOG(ERROR) << "Directory does not exist in ConvertReadTensors.";
      return;
    }
    ProcessConvertList(prefix_dump_file_name, file_format, specific_dump_dir, &dir_to_files_map, result_list);
    (void)closedir(d);
  }
  ConvertToHostFormat(dir_to_files_map, result_list);
}

void DebugServices::ConvertWatchPointNodes(const std::vector<std::tuple<std::string, std::string>> &proto_dump,
                                           const std::string &specific_dump_dir,
                                           std::vector<std::string> *const result_list) {
  std::string file_format = "npy";
  std::map<std::string, std::vector<std::string>> dir_to_files_map;
  for (const auto &node : proto_dump) {
    std::string dump_name = std::get<1>(node);
    dump_name = dump_name.substr(0, dump_name.rfind("."));
    // search files in dir for the one that meets the filename prefix and read the file into memory
    std::string abspath = RealPath(specific_dump_dir);
    DIR *d = opendir(abspath.c_str());
    if (d == nullptr) {
      MS_LOG(ERROR) << "Directory " << specific_dump_dir.c_str() << " does not exist in ConvertWatchPointNodes.";
      return;
    }
    ProcessConvertList(dump_name, file_format, specific_dump_dir, &dir_to_files_map, result_list);
    (void)closedir(d);
  }
  ConvertToHostFormat(dir_to_files_map, result_list);
}

void DebugServices::ProcessConvertList(const std::string &prefix_dump_file_name, const std::string &file_format,
                                       const std::string &specific_dump_dir,
                                       std::map<std::string, std::vector<std::string>> *dir_to_files_map,
                                       std::vector<std::string> *const result_list) {
  DIR *d = opendir(specific_dump_dir.c_str());
  struct dirent *dir = nullptr;
  while ((dir = readdir(d)) != nullptr) {
    if (dir->d_type != DT_REG) {
      continue;
    }
    std::string file_name = dir->d_name;
    std::string file_name_w_o_perfix = file_name;
    auto type_pos = file_name.find('.');
    if (type_pos == std::string::npos || file_name.find(prefix_dump_file_name, type_pos + 1) == std::string::npos) {
      continue;
    }
    if (file_name.rfind(file_format) == std::string::npos) {
      // if file matches prefix and is in device format add to candidate files to convert.
      (*dir_to_files_map)[specific_dump_dir].push_back(file_name);
    } else {
      // otherwise, if file matches prefix and already has been converted to host format
      // add to result of converted files.
      std::string found_file = specific_dump_dir + "/" + file_name;
      if (std::find(result_list->begin(), result_list->end(), found_file) == result_list->end()) {
        result_list->push_back(found_file);
      }
    }
  }
  (void)closedir(d);
}

void DebugServices::GetTensorDataInfoAsync(const std::vector<std::tuple<std::string, std::string>> &proto_dump,
                                           const std::string &specific_dump_dir, uint32_t iteration, uint32_t device_id,
                                           uint32_t root_graph_id, const std::vector<std::string> &async_file_pool,
                                           std::vector<std::shared_ptr<TensorData>> *const tensor_list) {
  for (auto &node : proto_dump) {
    std::vector<size_t> slot_list;
    std::string dump_style_name = std::get<1>(node);
    // Get dump_name and output_str from the second element of tuple
    std::size_t found_dot = dump_style_name.rfind(".");
    std::string dump_name = dump_style_name.substr(0, found_dot);
    std::string output_str = dump_style_name.substr(found_dot + 1);
    bool output_flag = (output_str == "output");

    for (const std::string &file_name : async_file_pool) {
      std::size_t found = file_name.find(dump_name);
      std::size_t found_out = file_name.find(output_str);
      std::size_t found_dot_start = file_name.find(".", found_out);
      std::size_t found_dot_end = file_name.find(".", found_dot_start);

      if (file_name.find(specific_dump_dir) != std::string::npos && found != std::string::npos &&
          found_out != std::string::npos) {
        slot_list.push_back(std::stoul(file_name.substr(found_dot_start + 1, found_dot_end - found_dot_start - 1)));
      }
    }
    for (auto slot : slot_list) {
      // add a TensorData entry (data will be read when needed)
      std::vector<int64_t> shape;
      std::string orig_name = std::get<0>(node);
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

      tensor_list->push_back(tensor_data);
    }
  }
}

void DebugServices::AddToTensorData(const std::string &backend_name, const std::string &time_stamp,
                                    const std::size_t slot, const unsigned int iteration, const unsigned int device_id,
                                    const unsigned int root_graph_id, const bool is_output, const std::size_t data_size,
                                    const std::string &type_name, const std::vector<int64_t> &shape,
                                    std::vector<char> *buffer,
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
    tensor_data->SetDataPtr(buffer->data());
  } else {
    tensor_data->SetDataPtr(nullptr);
  }
  tensor_data->SetByteSize(data_size);
  tensor_data->SetType(type_name);
  tensor_data->SetShape(shape);
  tensor_data->SetTimeStamp(time_stamp);
  if (data_size) {
    (void)tensor_loader_->LoadNewTensor(tensor_data, false);
  }

  // add to result_list
  result_list->push_back(tensor_data);
}

void DebugServices::SetPrefixToCheck(std::string *const prefix_dump_file_name, std::string *const slot_string_to_check,
                                     std::string *const dump_style_kernel_name, size_t slot, bool is_output) {
  std::string dump_style_name_part = *dump_style_kernel_name;
  dump_style_name_part = GetNodeNameWithoutScope(dump_style_name_part);
  std::string slot_str;
  if (is_output) {
    slot_str = ".output." + std::to_string(slot);
  } else {
    slot_str = ".input." + std::to_string(slot);
  }
  dump_style_name_part += slot_str;
  *prefix_dump_file_name = dump_style_name_part;
  *slot_string_to_check = slot_str;
}

std::string GetNewestFilePath(std::vector<std::string> file_list) {
  // get file with the newest timestamp from the list.
  if (file_list.empty()) {
    return "";
  }
  std::sort(file_list.begin(), file_list.end());
  return file_list.back();
}

std::string GetTimeStampStr(std::string file_path) {
  // get the file_name from file_path.
  size_t pos = file_path.rfind("/");
  std::string file_name = file_path.substr(pos + 1);
  size_t first_dot = file_name.rfind(".");
  size_t second_dot = file_name.rfind(".", first_dot - 1);
  size_t third_dot = file_name.rfind(".", second_dot - 1);
  size_t fourth_dot = file_name.rfind(".", third_dot - 1);
  size_t fifth_dot = file_name.rfind(".", fourth_dot - 1);
  if (fourth_dot != std::string::npos && fifth_dot != std::string::npos && fourth_dot > fifth_dot) {
    std::string time_stamp = file_name.substr(fifth_dot + 1, fourth_dot - fifth_dot - 1);
    return time_stamp;
  }
  return "";
}

void DebugServices::ReadDumpedTensor(std::vector<std::string> backend_name, std::vector<size_t> slot,
                                     std::vector<unsigned int> device_id, std::vector<unsigned int> iteration,
                                     std::vector<unsigned int> root_graph_id, const std::vector<bool> &is_output,
                                     const std::vector<std::string> &async_file_pool,
                                     std::vector<std::shared_ptr<TensorData>> *const result_list,
                                     bool *no_mem_to_read) {
  for (unsigned int i = 0; i < backend_name.size(); i++) {
    // form prefix of the tensor file to read from graph pb node name
    std::string dump_style_kernel_name = backend_name[i];

    // remove slot from name
    std::size_t found_colon = dump_style_kernel_name.find_last_of(":");
    dump_style_kernel_name = dump_style_kernel_name.substr(0, found_colon);

    std::string slot_string_to_check;
    std::string prefix_dump_file_name;
    SetPrefixToCheck(&prefix_dump_file_name, &slot_string_to_check, &dump_style_kernel_name, slot[i], is_output[i]);
    std::string prefix_dump_to_check = GetNodeNameWithoutScope(dump_style_kernel_name);

    std::string specific_dump_dir = dump_dir_ + "/rank_" + std::to_string(device_id[i]) + "/" + net_name_ + "/" +
                                    std::to_string(root_graph_id[i]) + "/" + IterationString(iteration[i]);

    // search files in dir for the one that meets the filename prefix and read the file into memory
    if (is_sync_mode_) {
      ReadDumpedTensorSync(prefix_dump_file_name, specific_dump_dir, backend_name[i], slot[i], device_id[i],
                           iteration[i], root_graph_id[i], is_output[i], result_list, no_mem_to_read);
    } else {
      ReadDumpedTensorAsync(specific_dump_dir, prefix_dump_to_check, slot_string_to_check, backend_name[i], slot[i],
                            device_id[i], iteration[i], root_graph_id[i], is_output[i], async_file_pool, result_list,
                            no_mem_to_read);
    }
  }
}

void DebugServices::ReadFileAndAddToTensor(const bool found, const std::vector<std::string> &matched_paths,
                                           const std::string &backend_name, const unsigned int device_id,
                                           const unsigned int root_graph_id, const bool &is_output, size_t slot,
                                           bool *no_mem_to_read, unsigned int iteration,
                                           std::vector<std::shared_ptr<TensorData>> *result_list) {
  std::string time_stamp = "";
  std::string type_name = "";
  uint64_t data_size = 0;
  std::vector<int64_t> shape;
  std::vector<char> *buffer = nullptr;
  if (found) {
    std::string result_path = GetNewestFilePath(matched_paths);
    time_stamp = GetTimeStampStr(result_path);
    std::string key_name_in_cache = backend_name + ":" + std::to_string(device_id) + ":" +
                                    std::to_string(root_graph_id) + ":" + std::to_string(is_output) + ":" +
                                    std::to_string(slot);
    ReadTensorFromNpy(key_name_in_cache, result_path, &type_name, &data_size, &shape, &buffer, no_mem_to_read);
    AddToTensorData(backend_name, time_stamp, slot, iteration, device_id, root_graph_id, is_output, data_size,
                    type_name, shape, buffer, result_list);
  } else {
    AddToTensorData(backend_name, time_stamp, slot, iteration, device_id, root_graph_id, is_output, 0, type_name, shape,
                    buffer, result_list);
    MS_LOG(INFO) << "Target tensor has not been found.";
  }
}

void DebugServices::ReadDumpedTensorSync(const std::string &prefix_dump_file_name, const std::string &specific_dump_dir,
                                         const std::string &backend_name, size_t slot, const unsigned int device_id,
                                         unsigned int iteration, unsigned int root_graph_id, const bool &is_output,
                                         std::vector<std::shared_ptr<TensorData>> *result_list, bool *no_mem_to_read) {
  std::string abspath = RealPath(specific_dump_dir);
  DIR *d = opendir(abspath.c_str());
  bool found_file = false;
  std::vector<std::string> matched_paths;
  if (d == nullptr) {
    MS_LOG(INFO) << "Directory " << specific_dump_dir << " does not exist!";
  } else {
    struct dirent *dir = nullptr;
    while ((dir = readdir(d)) != nullptr) {
      if (dir->d_type == DT_REG) {
        std::string file_name = dir->d_name;
        std::string stripped_file_name = GetStrippedFilename(file_name);
        if (stripped_file_name.empty()) {
          continue;
        }
        std::size_t found = stripped_file_name.rfind(prefix_dump_file_name, 0);
        if (found != 0) {
          continue;
        }
        std::string full_path = specific_dump_dir + "/" + file_name;
        matched_paths.push_back(full_path);
        found_file = true;
      }
    }
    (void)closedir(d);
  }
  ReadFileAndAddToTensor(found_file, matched_paths, backend_name, device_id, root_graph_id, is_output, slot,
                         no_mem_to_read, iteration, result_list);
}

void DebugServices::ReadDumpedTensorAsync(const std::string &specific_dump_dir, const std::string &prefix_dump_to_check,
                                          const std::string &slot_string_to_check, const std::string &backend_name,
                                          size_t slot, unsigned int device_id, unsigned int iteration,
                                          unsigned int root_graph_id, const bool &is_output,
                                          const std::vector<std::string> &async_file_pool,
                                          std::vector<std::shared_ptr<TensorData>> *result_list, bool *no_mem_to_read) {
  bool found = false;
  std::vector<std::string> matched_paths;
  // if async mode
  for (const std::string &file_path : async_file_pool) {
    if (file_path.find(specific_dump_dir) != std::string::npos &&
        file_path.find(prefix_dump_to_check) != std::string::npos &&
        file_path.find(slot_string_to_check) != std::string::npos) {
      matched_paths.push_back(file_path);
      found = true;
    }
  }
  ReadFileAndAddToTensor(found, matched_paths, backend_name, device_id, root_graph_id, is_output, slot, no_mem_to_read,
                         iteration, result_list);
}

std::string DebugServices::GetStrippedFilename(const std::string &file_name) {
  // strip off the task_id, stream_id, and timestamp, then compare
  size_t first_dot = file_name.find(".");
  size_t seventh_dot = file_name.rfind(".", file_name.rfind(".") - 1);
  size_t fifth_dot = file_name.rfind(".", file_name.rfind(".", seventh_dot - 1) - 1);

  if (fifth_dot == std::string::npos || fifth_dot <= first_dot) {
    return std::string();
  }

  // Look for the second dot's position from the back to avoid issue due to dots in the node name.
  size_t second_dot = fifth_dot;
  const int8_t kSecondDotPosition = 2;
  for (int8_t pos = 5; pos > kSecondDotPosition; pos--) {
    second_dot = file_name.rfind(".", second_dot - 1);
  }

  if (second_dot == std::string::npos || second_dot <= first_dot) {
    return std::string();
  }

  std::string start_string = file_name.substr(first_dot + 1, second_dot - first_dot - 1);
  std::string end_string = file_name.substr(fifth_dot, seventh_dot - fifth_dot);
  std::string stripped_file_name = start_string + end_string;
  return stripped_file_name;
}

std::vector<std::shared_ptr<TensorData>> DebugServices::ReadNeededDumpedTensors(
  unsigned int iteration, std::vector<std::string> *const async_file_pool) {
  // get a list of nodes and the devices they are on to monitor
  std::vector<std::shared_ptr<TensorData>> tensor_list;
  std::map<std::tuple<uint32_t, uint32_t>, std::vector<std::tuple<std::string, bool>>> device_and_graph_to_nodes;
  for (auto w_table_item : watchpoint_table_) {
    auto wp = std::get<1>(w_table_item);
    unsigned int index = 0;
    for (auto check_node : wp.check_node_list) {
      std::vector<uint32_t> devices = std::get<1>(wp.check_node_device_list[index]);
      std::vector<uint32_t> graphs = std::get<1>(wp.check_node_graph_list[index]);
      for (auto device : devices) {
        for (auto graph : graphs) {
          std::tuple<uint32_t, uint32_t> key(device, graph);
          device_and_graph_to_nodes[key].push_back(check_node);
        }
      }

      index++;
    }
  }

  // scan each device/iteration dir for the watched nodes for each device, and add to tensor_list
  // as they are found
  for (auto const &device_and_graph_item : device_and_graph_to_nodes) {
    std::tuple<uint32_t, uint32_t> device_and_graph = device_and_graph_item.first;
    uint32_t device_id = std::get<0>(device_and_graph);
    uint32_t root_graph_id = std::get<1>(device_and_graph);
    std::vector<std::tuple<std::string, bool>> wp_nodes = device_and_graph_item.second;
    std::vector<std::tuple<std::string, std::string>> proto_to_dump;

    std::string specific_dump_dir = dump_dir_ + "/rank_" + std::to_string(device_id) + "/" + net_name_ + "/" +
                                    std::to_string(root_graph_id) + "/" + IterationString(iteration);

    // convert node names to dump style
    for (auto node : wp_nodes) {
      std::string orig_name = std::get<0>(node);
      // Remove the scope from the fully qualified name to compare for both sync and async case.
      std::string dump_style_name = GetNodeNameWithoutScope(orig_name);

      bool node_is_out = std::get<1>(node);
      if (node_is_out) {
        dump_style_name += ".output";
      } else {
        dump_style_name += ".input";
      }
      if (std::find(proto_to_dump.begin(), proto_to_dump.end(),
                    std::tuple<std::string, std::string>(orig_name, dump_style_name)) == proto_to_dump.end()) {
        proto_to_dump.push_back(std::tuple<std::string, std::string>(orig_name, dump_style_name));
      }
    }

    if (is_sync_mode_) {
      std::string abspath = RealPath(specific_dump_dir);
      ProcessTensorDataSync(proto_to_dump, abspath, specific_dump_dir, iteration, device_id, root_graph_id,
                            &tensor_list);
    } else {
      // convert all files in proto_to_dump to npy and add to pool of async file names
      ConvertWatchPointNodes(proto_to_dump, specific_dump_dir, async_file_pool);
      GetTensorDataInfoAsync(proto_to_dump, specific_dump_dir, iteration, device_id, root_graph_id, *async_file_pool,
                             &tensor_list);
    }
  }

  return tensor_list;
}

void DebugServices::ProcessTensorDataSync(const std::vector<std::tuple<std::string, std::string>> &proto_to_dump,
                                          const std::string &abspath, const std::string &specific_dump_dir,
                                          unsigned int iteration, unsigned int device_id, unsigned int root_graph_id,
                                          std::vector<std::shared_ptr<TensorData>> *const tensor_list) {
  DIR *d = opendir(abspath.c_str());
  if (d == nullptr) {
    MS_LOG(ERROR) << "Directory " << specific_dump_dir.c_str() << " does not exist in ReadNeededDumpedTensors.";
  } else {
    struct dirent *dir = nullptr;
    while ((dir = readdir(d)) != nullptr) {
      if (dir->d_type == DT_REG) {
        std::string file_name = dir->d_name;
        for (auto &node : proto_to_dump) {
          std::string dump_name = std::get<1>(node);

          std::string stripped_file_name = GetStrippedFilename(file_name);
          if (stripped_file_name.empty() || stripped_file_name.length() <= dump_name.length()) {
            continue;
          }
          std::size_t found = stripped_file_name.rfind(dump_name, 0);
          if (found == 0) {
            size_t slot = std::stoul(stripped_file_name.substr(dump_name.length() + 1));
            std::vector<int64_t> shape;
            std::string orig_name = std::get<0>(node);
            std::string output_str = dump_name.substr(dump_name.rfind(".") + 1);
            bool output_flag = (output_str == "output");

            AddToTensorData(orig_name, "", slot, iteration, device_id, root_graph_id, output_flag, 0, "", shape,
                            nullptr, tensor_list);
            break;
          }
        }
      }
    }
    (void)closedir(d);
  }
}

std::string DebugServices::IterationString(unsigned int iteration) {
  std::string iteration_string;
  bool init_dbg_suspend = (iteration == UINT_MAX);
  if (init_dbg_suspend) {
    iteration_string = "init";
  } else {
    iteration_string = std::to_string(iteration);
  }
  return iteration_string;
}
#endif

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
    (void)ret_name->emplace_back(std::get<0>(result));
    (void)data_ptr->emplace_back(reinterpret_cast<const char *>(std::get<1>(result)->GetDataPtr()));
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
    auto input_size = AnfAlgo::GetInputTensorNum(kernel);
    for (size_t j = 0; j < input_size; ++j) {
      auto input_kernel = kernel->input(j + 1);
      std::string input_kernel_name = GetKernelNodeName(input_kernel);
      auto found = w_name.find_last_of('/');
      if (found != std::string::npos && w_name.size() > found && w_name.substr(found + 1) == input_kernel_name)
        return true;
    }
    return false;
  } else {
    return false;
  }
}
#endif

std::vector<std::shared_ptr<TensorData>> DebugServices::GetTensor() const { return tensor_loader_->GetTensor(); }

void DebugServices::EmptyCurrentTensor() { tensor_loader_->EmptyCurrentTensor(); }

#ifdef ONLINE_DBG_MODE
bool DebugServices::DumpTensorToFile(const std::string &tensor_name, bool trans_flag, const std::string &filepath,
                                     const std::string &host_fmt, const std::vector<int64_t> &host_shape,
                                     TypeId host_type, TypeId device_type, const std::string &addr_format,
                                     size_t slot) const {
  return tensor_loader_->DumpTensorToFile(tensor_name, trans_flag, filepath, host_fmt, host_shape, host_type,
                                          device_type, addr_format, slot);
}
#endif

bool DebugServices::LoadNewTensor(const std::shared_ptr<TensorData> &tensor, bool keep_prev) {
  return tensor_loader_->LoadNewTensor(tensor, keep_prev);
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

bool DebugServices::CheckOpOverflow(std::string node_name_to_find, unsigned int device_id, unsigned int root_graph_id,
                                    unsigned int iteration) {
  std::replace(node_name_to_find.begin(), node_name_to_find.end(), '/', '_');
  std::vector<std::string> op_names;
  std::string overflow_bin_path;

#ifdef ONLINE_DBG_MODE
  if (DumpJsonParser::GetInstance().path().empty()) {
    // Dump config is not set.
    return false;
  }
  auto debugger = Debugger::GetInstance();
  overflow_bin_path = DumpJsonParser::GetInstance().GetOpOverflowBinPath(debugger->GetGraphPtr()->root_graph_id());
  auto realpath = FileUtils::GetRealPath(overflow_bin_path.c_str());
  if (!realpath.has_value()) {
    MS_LOG(INFO) << "Get real path failed for overflow_bin_path.";
    return false;
  }
  overflow_bin_path = realpath.value() + '/';
#else
  overflow_bin_path = dump_dir_ + "/rank_" + std::to_string(device_id) + "/" + net_name_ + "/" +
                      std::to_string(root_graph_id) + "/" + IterationString(iteration) + "/";
  overflow_bin_path = RealPath(overflow_bin_path);
#endif

  overflow_wp_lock_.lock();

  MS_LOG(INFO) << "Searching for overflow in node " << node_name_to_find;
  auto found_overflows = overflow_ops_.find(overflow_bin_path);
  if (found_overflows != overflow_ops_.end()) {
    MS_LOG(INFO) << "Found already computed overflows for " << overflow_bin_path;
    op_names = overflow_ops_[overflow_bin_path];
  } else {
    std::map<std::pair<uint64_t, uint64_t>, std::string> task_stream_to_opname;
    std::vector<std::pair<uint64_t, uint64_t>> task_stream_hit;
    const std::string overflow_file_prefix = "Opdebug.Node_OpDebug.";

    MS_LOG(INFO) << "Processing bin file path " << overflow_bin_path;

    std::string abspath = RealPath(overflow_bin_path);
    DIR *d = opendir(abspath.c_str());
    if (d == nullptr) {
      MS_LOG(ERROR) << "OverFlow bin directory does not exist!";
    } else {
      struct dirent *dir = nullptr;
      while ((dir = readdir(d)) != nullptr) {
        if (dir->d_type == DT_REG) {
          // form fully qualified  filename
          std::string file_path = overflow_bin_path;
          std::string file_name = dir->d_name;
          (void)file_path.append(file_name);
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
          // detect overflow bin file
          if (file_name.rfind(overflow_file_prefix, 0) == 0) {
            // start of op overflow data in bin file
            const uint32_t offset = 321;
            (void)infile.seekg(offset, std::ios::beg);
            std::vector<char> buffer;
            // size of op overflow info section
            const size_t buf_size = 256;
            buffer.resize(buf_size);
            (void)infile.read(buffer.data(), buf_size);
            if (infile.gcount() != buf_size) {
              MS_LOG(ERROR) << "The file: " << file_path << "may be damaged!";
              continue;
            }
            const uint8_t stream_id_offset = 16;
            const uint8_t task_id_offset = 24;
            // The stream_id and task_id in the dump file are 8 byte fields for extensibility purpose, but only hold 4
            // byte values currently.
            stream_id = BytestoUInt64(std::vector<char>(buffer.begin() + stream_id_offset, buffer.end()));
            task_id = BytestoUInt64(std::vector<char>(buffer.begin() + task_id_offset, buffer.end()));
            MS_LOG(INFO) << "Overflow bin file " << file_name << ", task_id " << task_id << ", stream_id " << stream_id
                         << ".";
            task_stream_hit.push_back(std::make_pair(task_id, stream_id));
          } else {
            // regular bin file
            bool success_parse = GetAttrsFromAsyncFilename(file_name, &node_name, &task_id, &stream_id);
            if (success_parse) {
              task_stream_to_opname[std::make_pair(task_id, stream_id)] = node_name;
            }
          }
          infile.close();
        }
      }
      (void)closedir(d);
    }

    // find the op_names with an overflow hit
    for (auto &task_stream : task_stream_hit) {
      auto op_name = task_stream_to_opname[task_stream];
      if (!op_name.empty()) {
        MS_LOG(INFO) << "Operation overflow detected in " << op_name;
        op_names.push_back(op_name);
      }
    }

    overflow_ops_[overflow_bin_path] = op_names;
  }

  overflow_wp_lock_.unlock();

  // determine if overflow wp has been triggered for node_name_to_find
  if (find(op_names.begin(), op_names.end(), node_name_to_find) != op_names.end()) {
    MS_LOG(INFO) << "Operation overflow watchpoint triggered for  " << node_name_to_find;
    return true;
  }

  return false;
}

bool DebugServices::GetAttrsFromAsyncFilename(const std::string &file_name, std::string *const node_name,
                                              uint64_t *task_id, uint64_t *stream_id) {
  // get the node_name, task_id, and stream_id from async dump filename
  // node_type.node_name.task_id.stram_id.timestamp
  // WARNING: node_name may have dots in it
  size_t fourth_dot = file_name.rfind(".");
  size_t third_dot = file_name.rfind(".", fourth_dot - 1);
  size_t second_dot = file_name.rfind(".", third_dot - 1);
  size_t first_dot = file_name.find(".");

  // check if dots were found
  if (first_dot == std::string::npos || second_dot == std::string::npos || third_dot == std::string::npos ||
      fourth_dot == std::string::npos) {
    return false;
  }

  // check if its not an async bin file
  if (file_name.substr(fourth_dot) == ".npy") {
    return false;
  }

  // get node_name
  if (first_dot < second_dot) {
    *node_name = file_name.substr(first_dot + 1, second_dot - first_dot - 1);
  } else {
    MS_LOG(ERROR) << "Async filename parse error to get node_name.";
    return false;
  }

  // get task id
  if (second_dot < third_dot) {
    std::string extracted_task_id = file_name.substr(second_dot + 1, third_dot - second_dot - 1);
    try {
      *task_id = std::stoull(extracted_task_id);
    } catch (std::invalid_argument &e) {
      MS_LOG(ERROR) << "stoull failed on extracted_task_id to get task_id, invalid argument.";
      return false;
    } catch (std::out_of_range &e) {
      MS_LOG(ERROR) << "stoull failed on extracted_task_id to get task_id, out of range.";
      return false;
    }
  } else {
    MS_LOG(ERROR) << "Async filename parse error to get task_id.";
    return false;
  }

  // get stream id
  if (third_dot < fourth_dot) {
    std::string extracted_stream_id = file_name.substr(third_dot + 1, fourth_dot - third_dot - 1);
    try {
      *stream_id = std::stoull(extracted_stream_id);
    } catch (std::invalid_argument &e) {
      MS_LOG(ERROR) << "stoull failed on extracted_stream_id to get stream_id, invalid argument.";
      return false;
    } catch (std::out_of_range &e) {
      MS_LOG(ERROR) << "stoull failed on extracted_stream_id to get stream_id, out of range.";
      return false;
    }
  } else {
    MS_LOG(ERROR) << "Async filename parse error to get stream_id.";
    return false;
  }

  return true;
}

std::string DebugServices::RealPath(const std::string &input_path) {
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
      MS_LOG(ERROR) << "The dir " << prefix_path << " does not exist.";
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

uint64_t DebugServices::BytestoUInt64(const std::vector<char> &buffer) {
  return le64toh(*reinterpret_cast<const uint64_t *>(buffer.data()));
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

bool DebugServices::GetSyncMode() { return is_sync_mode_; }

void DebugServices::SetMemLimit(uint64_t max_mem_size) { tensor_loader_->SetMemTotal(max_mem_size); }

#ifdef ONLINE_DBG_MODE
}  // namespace mindspore
#endif
