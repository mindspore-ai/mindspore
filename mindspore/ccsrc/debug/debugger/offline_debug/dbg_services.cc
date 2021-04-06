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

DbgServices::DbgServices(bool verbose) {
  DbgLogger::verbose = verbose;
  char *dbg_log_path = getenv("OFFLINE_DBG_LOG");
  if (dbg_log_path != NULL) {
    DbgLogger::verbose = true;
  }
  debug_services = new DebugServices();
}

DbgServices::DbgServices(const DbgServices &other) {
  MS_LOG(INFO) << "cpp DbgServices object is created via copy";
  debug_services = new DebugServices(*other.debug_services);
}

DbgServices &DbgServices::operator=(const DbgServices &other) {
  MS_LOG(INFO) << "cpp DbgServices object is being assigned a different state";
  if (this != &other) {
    delete debug_services;
    debug_services = new DebugServices(*other.debug_services);
  }
  return *this;
}

DbgServices::~DbgServices() {
  MS_LOG(INFO) << "cpp DbgServices object is deleted";
  delete debug_services;
}

std::string DbgServices::GetVersion() {
  MS_LOG(INFO) << "get version is called";
  return "1.2.0";
}

int32_t DbgServices::Initialize(std::string net_name, std::string dump_folder_path, bool is_sync_mode) {
  MS_LOG(INFO) << "cpp DbgServices initialize network name " << net_name;
  MS_LOG(INFO) << "cpp DbgServices initialize dump folder path " << dump_folder_path;
  MS_LOG(INFO) << "cpp DbgServices initialize sync mode " << is_sync_mode;
  debug_services->SetNetName(net_name);
  debug_services->SetDumpDir(dump_folder_path);
  debug_services->SetSyncMode(is_sync_mode);
  return 0;
}

int32_t DbgServices::AddWatchpoint(
  unsigned int id, unsigned int watch_condition,
  std::map<std::string, std::map<std::string, std::variant<bool, std::vector<std::string>>>> check_nodes,
  std::vector<parameter_t> parameter_list) {
  MS_LOG(INFO) << "cpp start";

  MS_LOG(INFO) << "cpp DbgServices AddWatchpoint id " << id;
  MS_LOG(INFO) << "cpp DbgServices AddWatchpoint watch_condition " << watch_condition;
  for (auto const &node : check_nodes) {
    MS_LOG(INFO) << "cpp DbgServices AddWatchpoint name " << node.first;
    auto attr_map = node.second;

    bool is_parameter = std::get<bool>(attr_map["is_parameter"]);
    MS_LOG(INFO) << "cpp DbgServices AddWatchpoint is_parameter " << is_parameter;

    // std::vector<uint32_t> device_id = std::get<std::vector<uint32_t>>(attr_map["device_id"]);
    std::vector<std::string> device_id_str = std::get<std::vector<std::string>>(attr_map["device_id"]);
    std::vector<std::uint32_t> device_id;
    std::transform(device_id_str.begin(), device_id_str.end(), std::back_inserter(device_id),
                   [](std::string &id_str) -> std::uint32_t { return static_cast<uint32_t>(std::stoul(id_str)); });
    MS_LOG(INFO) << "cpp DbgServices AddWatchpoint device_id ";
    for (auto const &i : device_id) {
      MS_LOG(INFO) << i << " ";
    }

    // std::vector<uint32_t> root_graph_id = std::get<std::vector<uint32_t>>(attr_map["root_graph_id"]);
    std::vector<std::string> root_graph_id_str = std::get<std::vector<std::string>>(attr_map["root_graph_id"]);
    std::vector<std::uint32_t> root_graph_id;
    std::transform(
      root_graph_id_str.begin(), root_graph_id_str.end(), std::back_inserter(root_graph_id),
      [](std::string &graph_str) -> std::uint32_t { return static_cast<uint32_t>(std::stoul(graph_str)); });
    MS_LOG(INFO) << "cpp DbgServices AddWatchpoint root_graph_id";
    for (auto const &j : root_graph_id) {
      MS_LOG(INFO) << j << " ";
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

  std::transform(check_nodes.begin(), check_nodes.end(), std::back_inserter(check_node_list),
                 [](auto &node) -> std::tuple<std::string, bool> {
                   auto attr_map = node.second;
                   return std::make_tuple(node.first, std::get<bool>(attr_map["is_parameter"]));
                 });

  std::transform(check_nodes.begin(), check_nodes.end(), std::back_inserter(check_node_device_list),
                 [](auto &node) -> std::tuple<std::string, std::vector<uint32_t>> {
                   auto attr_map = node.second;
                   std::vector<std::string> device_id_str = std::get<std::vector<std::string>>(attr_map["device_id"]);
                   std::vector<std::uint32_t> device_id;
                   std::transform(
                     device_id_str.begin(), device_id_str.end(), std::back_inserter(device_id),
                     [](std::string &id_str) -> std::uint32_t { return static_cast<uint32_t>(std::stoul(id_str)); });
                   return std::make_tuple(node.first, device_id);
                 });

  std::transform(
    check_nodes.begin(), check_nodes.end(), std::back_inserter(check_node_graph_list),
    [](auto &node) -> std::tuple<std::string, std::vector<uint32_t>> {
      auto attr_map = node.second;
      std::vector<std::string> root_graph_id_str = std::get<std::vector<std::string>>(attr_map["root_graph_id"]);
      std::vector<std::uint32_t> root_graph_id;
      std::transform(
        root_graph_id_str.begin(), root_graph_id_str.end(), std::back_inserter(root_graph_id),
        [](std::string &graph_str) -> std::uint32_t { return static_cast<uint32_t>(std::stoul(graph_str)); });
      return std::make_tuple(node.first, root_graph_id);
    });

  std::transform(
    parameter_list.begin(), parameter_list.end(), std::back_inserter(parameter_list_backend),
    [](const parameter_t &parameter) -> DebugServices::parameter_t {
      return DebugServices::parameter_t{parameter.name, parameter.disabled, parameter.value, parameter.hit};
    });

  debug_services->AddWatchpoint(id, watch_condition, 0, check_node_list, parameter_list_backend,
                                &check_node_device_list, &check_node_graph_list);
  MS_LOG(INFO) << "cpp end";
  return 0;
}

int32_t DbgServices::RemoveWatchpoint(unsigned int id) {
  MS_LOG(INFO) << "cpp DbgServices RemoveWatchpoint id " << id;
  debug_services->RemoveWatchpoint(id);
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
  std::vector<unsigned int> device_id;
  std::vector<unsigned int> root_graph_id;
  // #ifdef ENABLE_D
  //  overflow_ops = CheckOpOverflow();
  // #endif

  std::vector<std::shared_ptr<TensorData>> tensor_list;
  tensor_list = debug_services->ReadNeededDumpedTensors(iteration);

  debug_services->CheckWatchpoints(&name, &slot, &condition, &watchpoint_id, &parameters, &error_codes, overflow_ops,
                                   &tensor_list, false, true, true, &device_id, &root_graph_id);

  std::vector<watchpoint_hit_t> hits;
  for (unsigned int i = 0; i < name.size(); i++) {
    std::vector<DebugServices::parameter_t> &parameter = parameters[i];
    std::vector<parameter_t> api_parameter_vector;
    for (const auto &p : parameter) {
      parameter_t api_parameter(p.name, p.disabled, p.value, p.hit, p.actual_value);
      api_parameter_vector.push_back(api_parameter);
    }
    watchpoint_hit_t hit(name[i], std::stoi(slot[i]), condition[i], watchpoint_id[i], api_parameter_vector,
                         error_codes[i], device_id[i], root_graph_id[i]);

    MS_LOG(INFO) << "cpp DbgServices watchpoint_hit_t name " << hit.name;
    MS_LOG(INFO) << "cpp DbgServices watchpoint_hit_t slot " << hit.slot;
    MS_LOG(INFO) << "cpp DbgServices watchpoint_hit_t watchpoint_id " << hit.watchpoint_id;
    MS_LOG(INFO) << "cpp DbgServices watchpoint_hit_t error_code " << hit.error_code;
    MS_LOG(INFO) << "cpp DbgServices watchpoint_hit_t device_id " << hit.device_id;
    MS_LOG(INFO) << "cpp DbgServices watchpoint_hit_t root_graph_id " << hit.root_graph_id;

    for (auto const &parameter_i : api_parameter_vector) {
      MS_LOG(INFO) << "cpp DbgServices watchpoint_hit_t parameter name " << parameter_i.name;
      MS_LOG(INFO) << "cpp DbgServices watchpoint_hit_t parameter disabled " << parameter_i.disabled;
      MS_LOG(INFO) << "cpp DbgServices watchpoint_hit_t parameter value " << parameter_i.value;
      MS_LOG(INFO) << "cpp DbgServices watchpoint_hit_t parameter hit " << parameter_i.hit;
      MS_LOG(INFO) << "cpp DbgServices watchpoint_hit_t parameter actual_value " << parameter_i.actual_value;
    }

    hits.push_back(hit);
  }
  return hits;
}

std::string GetTensorFullName(tensor_info_t info) {
  std::string node_name = info.node_name;
  if (info.is_parameter) {
    // scopes in node name are separated by '/'
    // use the name without scope if truncate is true
    std::size_t found = node_name.find_last_of("/");
    node_name = node_name.substr(found + 1);
  }
  return node_name + ":" + std::to_string(info.slot);
}

unsigned int GetTensorDeviceId(tensor_info_t info) { return info.device_id; }

unsigned int GetTensorRootGraphId(tensor_info_t info) { return info.root_graph_id; }

unsigned int GetTensorIteration(tensor_info_t info) { return info.iteration; }

unsigned int GetTensorSlot(tensor_info_t info) { return info.slot; }

std::vector<tensor_data_t> DbgServices::ReadTensors(std::vector<tensor_info_t> info) {
  for (auto i : info) {
    MS_LOG(INFO) << "cpp DbgServices ReadTensor info name " << i.node_name << ", slot " << i.slot << ", iteration "
                 << i.iteration << ", device_id " << i.device_id << ", root_graph_id " << i.root_graph_id;
  }
  std::vector<std::string> backend_name;
  std::vector<unsigned int> device_id;
  std::vector<unsigned int> root_graph_id;
  std::vector<unsigned int> iteration;
  std::vector<size_t> slot;
  std::vector<std::shared_ptr<TensorData>> result_list;
  std::vector<tensor_data_t> tensors_read;

  std::transform(info.begin(), info.end(), std::back_inserter(backend_name), GetTensorFullName);
  std::transform(info.begin(), info.end(), std::back_inserter(slot), GetTensorSlot);
  std::transform(info.begin(), info.end(), std::back_inserter(device_id), GetTensorDeviceId);
  std::transform(info.begin(), info.end(), std::back_inserter(root_graph_id), GetTensorRootGraphId);
  std::transform(info.begin(), info.end(), std::back_inserter(iteration), GetTensorIteration);

  MS_LOG(INFO) << "cpp before";
  debug_services->ReadDumpedTensor(backend_name, slot, device_id, iteration, root_graph_id, &result_list);
  MS_LOG(INFO) << "cpp after";

  for (auto result : result_list) {
    tensor_data_t tensor_data_item(result->GetDataPtr(), result->GetByteSize(), result->GetType(), result->GetShape());
    tensors_read.push_back(tensor_data_item);
  }
  MS_LOG(INFO) << "cpp end";
  return tensors_read;
}
