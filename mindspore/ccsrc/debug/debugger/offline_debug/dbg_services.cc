/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "debugger/offline_debug/dbg_services.h"

#include <algorithm>
#include <chrono>

DbgServices::DbgServices() { debug_services_ = std::make_shared<DebugServices>(); }

DbgServices::DbgServices(const DbgServices &other) {
  MS_LOG(INFO) << "cpp DbgServices object is created via copy";
  debug_services_ = other.debug_services_;
}

DbgServices &DbgServices::operator=(const DbgServices &other) {
  MS_LOG(INFO) << "cpp DbgServices object is being assigned a different state";
  if (this != &other) {
    debug_services_ = other.debug_services_;
  }
  return *this;
}

DbgServices::~DbgServices() noexcept {
  MS_LOG(INFO) << "cpp DbgServices object is deleted";
  debug_services_ = nullptr;
}

std::string DbgServices::GetVersion() const {
  MS_LOG(INFO) << "get version is called";
  return "1.5.0";
}

int32_t DbgServices::Initialize(const std::string net_name, const std::string dump_folder_path, bool is_sync_mode,
                                uint64_t max_mem_usage) {
  MS_LOG(INFO) << "cpp DbgServices initialize network name " << net_name;
  MS_LOG(INFO) << "cpp DbgServices initialize dump folder path " << dump_folder_path;
  MS_LOG(INFO) << "cpp DbgServices initialize sync mode " << is_sync_mode;
  MS_LOG(INFO) << "cpp DbgServices initialize maximum memory size for debugger internal cache " << max_mem_usage
               << "MB.";
  if (debug_services_ == nullptr) {
    MS_LOG(EXCEPTION) << "Debugger services initialize failed as occur null pointer error,"
                      << "may be due to memory allocation failure, check as: top";
  }
  debug_services_->SetNetName(net_name);
  debug_services_->SetDumpDir(dump_folder_path);
  debug_services_->SetSyncMode(is_sync_mode);
  // Set the memory ratio used by tensor cache. Leave 50% for other debugger backend usage.
  const uint64_t kMegabytesToBytes = 1048576;  // max_mem_usage will be bytes in unit in debugger backend.
  const uint64_t ratio_inversion = 2;
  const uint64_t memlimit = max_mem_usage * kMegabytesToBytes / ratio_inversion;
  debug_services_->SetMemLimit(memlimit);
  return 0;
}

int32_t DbgServices::AddWatchpoint(
  unsigned int id, unsigned int watch_condition,
  std::map<std::string, std::map<std::string, std::variant<bool, std::vector<std::string>>>> check_nodes,
  std::vector<parameter_t> parameter_list) {
  MS_LOG(INFO) << "cpp DbgServices start AddWatchpoint";

  MS_LOG(INFO) << "cpp DbgServices AddWatchpoint id " << id;
  MS_LOG(INFO) << "cpp DbgServices AddWatchpoint watch_condition " << watch_condition;
  for (auto const &node : check_nodes) {
    MS_LOG(DEBUG) << "cpp DbgServices AddWatchpoint name " << node.first;
    auto attr_map = node.second;

    bool is_output = std::get<bool>(attr_map["is_output"]);
    MS_LOG(DEBUG) << "cpp DbgServices AddWatchpoint is_output " << is_output;

    std::vector<std::string> rank_id_str = std::get<std::vector<std::string>>(attr_map["rank_id"]);
    std::vector<std::uint32_t> rank_id;
    (void)std::transform(
      rank_id_str.begin(), rank_id_str.end(), std::back_inserter(rank_id),
      [](std::string &id_str) -> std::uint32_t { return static_cast<uint32_t>(std::stoul(id_str)); });
    MS_LOG(DEBUG) << "cpp DbgServices AddWatchpoint rank_id: ";
    for (auto const &i : rank_id) {
      MS_LOG(DEBUG) << i << " ";
    }

    std::vector<std::string> root_graph_id_str = std::get<std::vector<std::string>>(attr_map["root_graph_id"]);
    std::vector<std::uint32_t> root_graph_id;
    (void)std::transform(
      root_graph_id_str.begin(), root_graph_id_str.end(), std::back_inserter(root_graph_id),
      [](std::string &graph_str) -> std::uint32_t { return static_cast<uint32_t>(std::stoul(graph_str)); });
    MS_LOG(DEBUG) << "cpp DbgServices AddWatchpoint root_graph_id: ";
    for (auto const &j : root_graph_id) {
      MS_LOG(DEBUG) << j << " ";
    }
  }

  for (auto const &parameter : parameter_list) {
    MS_LOG(INFO) << "cpp DbgServices AddWatchpoint parameter name " << parameter.name;
    MS_LOG(INFO) << "cpp DbgServices AddWatchpoint parameter disabled " << parameter.disabled;
    MS_LOG(INFO) << "cpp DbgServices AddWatchpoint parameter value " << parameter.value;
    MS_LOG(INFO) << "cpp DbgServices AddWatchpoint parameter hit " << parameter.hit;
    MS_LOG(INFO) << "cpp DbgServices AddWatchpoint parameter actual_value " << parameter.actual_value;
  }

  std::vector<std::tuple<std::string, bool>> check_node_list;
  std::vector<std::tuple<std::string, std::vector<uint32_t>>> check_node_device_list;
  std::vector<std::tuple<std::string, std::vector<uint32_t>>> check_node_graph_list;
  std::vector<DebugServices::parameter_t> parameter_list_backend;

  (void)std::transform(check_nodes.begin(), check_nodes.end(), std::back_inserter(check_node_list),
                       [](auto &node) -> std::tuple<std::string, bool> {
                         auto attr_map = node.second;
                         return std::make_tuple(node.first, std::get<bool>(attr_map["is_output"]));
                       });

  (void)std::transform(check_nodes.begin(), check_nodes.end(), std::back_inserter(check_node_device_list),
                       [](auto &node) -> std::tuple<std::string, std::vector<uint32_t>> {
                         auto attr_map = node.second;
                         std::vector<std::string> rank_id_str = std::get<std::vector<std::string>>(attr_map["rank_id"]);
                         std::vector<std::uint32_t> rank_id;
                         (void)std::transform(rank_id_str.begin(), rank_id_str.end(), std::back_inserter(rank_id),
                                              [](std::string &id_str) -> std::uint32_t {
                                                return static_cast<uint32_t>(std::stoul(id_str));
                                              });
                         return std::make_tuple(node.first, rank_id);
                       });

  (void)std::transform(
    check_nodes.begin(), check_nodes.end(), std::back_inserter(check_node_graph_list),
    [](auto &node) -> std::tuple<std::string, std::vector<uint32_t>> {
      auto attr_map = node.second;
      std::vector<std::string> root_graph_id_str = std::get<std::vector<std::string>>(attr_map["root_graph_id"]);
      std::vector<std::uint32_t> root_graph_id;
      (void)std::transform(
        root_graph_id_str.begin(), root_graph_id_str.end(), std::back_inserter(root_graph_id),
        [](std::string &graph_str) -> std::uint32_t { return static_cast<uint32_t>(std::stoul(graph_str)); });
      return std::make_tuple(node.first, root_graph_id);
    });

  (void)std::transform(
    parameter_list.begin(), parameter_list.end(), std::back_inserter(parameter_list_backend),
    [](const parameter_t &parameter) -> DebugServices::parameter_t {
      return DebugServices::parameter_t{parameter.name, parameter.disabled, parameter.value, parameter.hit};
    });

  debug_services_->AddWatchpoint(id, watch_condition, 0, check_node_list, parameter_list_backend,
                                 &check_node_device_list, &check_node_graph_list);
  MS_LOG(INFO) << "cpp DbgServices end AddWatchpoint";
  return 0;
}

int32_t DbgServices::RemoveWatchpoint(unsigned int id) {
  MS_LOG(INFO) << "cpp DbgServices RemoveWatchpoint id " << id;
  debug_services_->RemoveWatchpoint(id);
  return 0;
}

std::vector<watchpoint_hit_t> DbgServices::CheckWatchpoints(unsigned int iteration) {
  MS_LOG(INFO) << "cpp DbgServices CheckWatchpoint iteration " << iteration;

  std::vector<std::string> name;
  std::vector<std::string> slot;
  std::vector<int> condition;
  std::vector<unsigned int> watchpoint_id;
  std::vector<std::string> overflow_ops;
  std::vector<std::vector<DebugServices::parameter_t>> parameters;
  std::vector<int32_t> error_codes;
  std::vector<unsigned int> rank_id;
  std::vector<unsigned int> root_graph_id;
  std::vector<std::shared_ptr<TensorData>> tensor_list;
  std::vector<std::string> file_paths;

  const bool init_dbg_suspend = (iteration == UINT_MAX);

  tensor_list = debug_services_->ReadNeededDumpedTensors(iteration, &file_paths);

  debug_services_->CheckWatchpoints(&name, &slot, &condition, &watchpoint_id, &parameters, &error_codes, overflow_ops,
                                    file_paths, &tensor_list, init_dbg_suspend, true, true, &rank_id, &root_graph_id);

  std::vector<watchpoint_hit_t> hits;
  for (unsigned int i = 0; i < name.size(); i++) {
    std::vector<DebugServices::parameter_t> &parameter = parameters[i];
    std::vector<parameter_t> api_parameter_vector;
    for (const auto &p : parameter) {
      parameter_t api_parameter(p.name, p.disabled, p.value, p.hit, p.actual_value);
      api_parameter_vector.push_back(api_parameter);
    }
    watchpoint_hit_t hit(name[i], std::stoi(slot[i]), condition[i], watchpoint_id[i], api_parameter_vector,
                         error_codes[i], rank_id[i], root_graph_id[i]);

    MS_LOG(DEBUG) << "cpp DbgServices watchpoint_hit_t name " << hit.name;
    MS_LOG(DEBUG) << "cpp DbgServices watchpoint_hit_t slot " << hit.slot;
    MS_LOG(DEBUG) << "cpp DbgServices watchpoint_hit_t watchpoint_id " << hit.watchpoint_id;
    MS_LOG(DEBUG) << "cpp DbgServices watchpoint_hit_t error_code " << hit.error_code;
    MS_LOG(DEBUG) << "cpp DbgServices watchpoint_hit_t rank_id " << hit.rank_id;
    MS_LOG(DEBUG) << "cpp DbgServices watchpoint_hit_t root_graph_id " << hit.root_graph_id;

    for (auto const &parameter_i : api_parameter_vector) {
      MS_LOG(DEBUG) << "cpp DbgServices watchpoint_hit_t parameter name " << parameter_i.name;
      MS_LOG(DEBUG) << "cpp DbgServices watchpoint_hit_t parameter disabled " << parameter_i.disabled;
      MS_LOG(DEBUG) << "cpp DbgServices watchpoint_hit_t parameter value " << parameter_i.value;
      MS_LOG(DEBUG) << "cpp DbgServices watchpoint_hit_t parameter hit " << parameter_i.hit;
      MS_LOG(DEBUG) << "cpp DbgServices watchpoint_hit_t parameter actual_value " << parameter_i.actual_value;
    }

    hits.push_back(hit);
  }
  return hits;
}

std::string GetTensorFullName(const tensor_info_t info) { return info.node_name + ":" + std::to_string(info.slot); }

unsigned int GetTensorRankId(const tensor_info_t info) { return info.rank_id; }

unsigned int GetTensorRootGraphId(const tensor_info_t info) { return info.root_graph_id; }

unsigned int GetTensorIteration(const tensor_info_t info) { return info.iteration; }

unsigned int GetTensorSlot(const tensor_info_t info) { return info.slot; }

bool GetTensorIsOutput(const tensor_info_t info) { return info.is_output; }

std::vector<std::shared_ptr<TensorData>> DbgServices::ReadTensorsUtil(std::vector<tensor_info_t> info) {
  for (auto i : info) {
    MS_LOG(INFO) << "cpp DbgServices ReadTensor info name " << i.node_name << ", slot " << i.slot << ", iteration "
                 << i.iteration << ", rank_id " << i.rank_id << ", root_graph_id " << i.root_graph_id << ", is_output "
                 << i.is_output;
  }
  std::vector<std::string> backend_name;
  std::vector<unsigned int> rank_id;
  std::vector<unsigned int> root_graph_id;
  std::vector<unsigned int> iteration;
  std::vector<size_t> slot;
  std::vector<std::shared_ptr<TensorData>> result_list;
  std::vector<bool> is_output;

  (void)std::transform(info.begin(), info.end(), std::back_inserter(backend_name), GetTensorFullName);
  (void)std::transform(info.begin(), info.end(), std::back_inserter(slot), GetTensorSlot);
  (void)std::transform(info.begin(), info.end(), std::back_inserter(rank_id), GetTensorRankId);
  (void)std::transform(info.begin(), info.end(), std::back_inserter(root_graph_id), GetTensorRootGraphId);
  (void)std::transform(info.begin(), info.end(), std::back_inserter(iteration), GetTensorIteration);
  (void)std::transform(info.begin(), info.end(), std::back_inserter(is_output), GetTensorIsOutput);

  MS_LOG(INFO) << "cpp before";
  std::vector<std::string> file_paths;
  auto t1 = std::chrono::high_resolution_clock::now();
  // Convert the dumped data to npy format if it's async mode.
  if (!debug_services_->GetSyncMode()) {
    debug_services_->ConvertReadTensors(backend_name, slot, rank_id, iteration, root_graph_id, &file_paths);
  }
  debug_services_->ReadDumpedTensor(backend_name, slot, rank_id, iteration, root_graph_id, is_output, file_paths,
                                    &result_list);
  for (auto result : result_list) {
    std::string output = "0";
    if (result->GetIsOutput()) {
      output = "1";
    }
    std::string key_name_in_cache = result->GetName() + ":" + std::to_string(result->GetDeviceId()) + ":" +
                                    std::to_string(result->GetRootGraphId()) + ":" + output + ":" +
                                    std::to_string(result->GetSlot());
    debug_services_->AppendToCacheEvictQueue(key_name_in_cache);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  /* Getting number of milliseconds as a double. */
  std::chrono::duration<double, std::milli> ms_double = t2 - t1;

  MS_LOG(INFO) << "ReadTensors Took: " << ms_double.count() / 1000 << "s";
  MS_LOG(INFO) << "cpp after";

  return result_list;
}

std::vector<tensor_data_t> DbgServices::ReadTensors(const std::vector<tensor_info_t> info) {
  std::vector<tensor_data_t> tensors_read;
  std::vector<std::shared_ptr<TensorData>> result_list;
  result_list = ReadTensorsUtil(info);
  for (auto result : result_list) {
    tensor_data_t tensor_data_item(result->GetDataPtr(), result->GetByteSize(), result->GetType(), result->GetShape());
    tensors_read.push_back(tensor_data_item);
  }
  return tensors_read;
}

std::vector<TensorBaseData> DbgServices::ReadTensorsBase(const std::vector<tensor_info_t> info) {
  std::vector<TensorBaseData> tensors_read_base;
  std::vector<std::shared_ptr<TensorData>> result_list;
  result_list = ReadTensorsUtil(info);
  for (auto result : result_list) {
    if (!result->GetByteSize()) {
      // tensor not found, adding empty tensor base.
      TensorBaseData tensor_data_item(0, 0, {});
      tensors_read_base.push_back(tensor_data_item);
      continue;
    }
    TensorBaseData tensor_data_item(result->GetByteSize(), result->GetType(), result->GetShape());
    tensors_read_base.push_back(tensor_data_item);
  }
  return tensors_read_base;
}

void AddTensorStatInfo(const DebugServices::TensorStat &tensor_statistics,
                       std::vector<TensorStatData> *const tensors_read_stat) {
  if (tensors_read_stat == nullptr) {
    MS_LOG(DEBUG) << "tensors_read_stat is nullptr.";
    return;
  }
  TensorStatData tensor_data_item(
    tensor_statistics.data_size, tensor_statistics.dtype, tensor_statistics.shape, tensor_statistics.is_bool,
    tensor_statistics.max_value, tensor_statistics.min_value, tensor_statistics.avg_value, tensor_statistics.count,
    tensor_statistics.neg_zero_count, tensor_statistics.pos_zero_count, tensor_statistics.nan_count,
    tensor_statistics.neg_inf_count, tensor_statistics.pos_inf_count, tensor_statistics.zero_count);
  tensors_read_stat->push_back(tensor_data_item);
}

std::vector<TensorStatData> DbgServices::ReadTensorsStat(const std::vector<tensor_info_t> info) {
  std::vector<TensorStatData> tensors_read_stat;
  std::vector<std::shared_ptr<TensorData>> result_list;
  result_list = ReadTensorsUtil(info);
  for (auto result : result_list) {
    if (!result->GetByteSize()) {
      DebugServices::TensorStat tensor_statistics;
      AddTensorStatInfo(tensor_statistics, &tensors_read_stat);
      continue;
    }
    DebugServices::TensorStat tensor_statistics = debug_services_->GetTensorStatistics(result);
    AddTensorStatInfo(tensor_statistics, &tensors_read_stat);
  }

  return tensors_read_stat;
}
