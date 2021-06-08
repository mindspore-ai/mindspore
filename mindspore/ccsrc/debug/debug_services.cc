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
#include <iterator>
#include <map>
#include <numeric>
#include <unordered_set>
#include "pybind11/embed.h"
#ifdef ONLINE_DBG_MODE
#include "backend/session/anf_runtime_algorithm.h"
#endif
#include "debug/debugger/tensor_summary.h"
#ifdef ONLINE_DBG_MODE
namespace mindspore {
#endif
DebugServices::DebugServices() { tensor_loader_ = std::make_shared<TensorLoader>(); }

DebugServices::DebugServices(const DebugServices &other) {
  tensor_loader_ = other.tensor_loader_;
  watchpoint_table = other.watchpoint_table;
}

DebugServices &DebugServices::operator=(const DebugServices &other) {
  if (this != &other) {
    tensor_loader_ = other.tensor_loader_;
    watchpoint_table = other.watchpoint_table;
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
  watchpoint_table[id] = watchpoint_item;
}

void DebugServices::RemoveWatchpoint(unsigned int id) {
  std::lock_guard<std::mutex> lg(lock_);
  watchpoint_table.erase(id);
}

std::unique_ptr<ITensorSummary> GetSummaryPtr(const std::shared_ptr<TensorData> &tensor,
                                              void *const previous_tensor_ptr, uint32_t num_elements,
                                              int tensor_dtype) {
  switch (tensor_dtype) {
    case DbgDataType::DT_UINT8: {
      return std::make_unique<TensorSummary<uint8_t>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements);
    }
    case DbgDataType::DT_INT8: {
      return std::make_unique<TensorSummary<int8_t>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements);
    }
    case DbgDataType::DT_UINT16: {
      return std::make_unique<TensorSummary<uint16_t>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements);
    }
    case DbgDataType::DT_INT16: {
      return std::make_unique<TensorSummary<int16_t>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements);
    }
    case DbgDataType::DT_UINT32: {
      return std::make_unique<TensorSummary<uint32_t>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements);
    }
    case DbgDataType::DT_INT32:
    case DbgDataType::DT_BASE_INT: {
      return std::make_unique<TensorSummary<int32_t>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements);
    }
    case DbgDataType::DT_UINT64: {
      return std::make_unique<TensorSummary<uint64_t>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements);
    }
    case DbgDataType::DT_INT64: {
      return std::make_unique<TensorSummary<int64_t>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements);
    }
    case DbgDataType::DT_FLOAT16: {
      return std::make_unique<TensorSummary<float16>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements);
    }
    case DbgDataType::DT_FLOAT32:
    case DbgDataType::DT_BASE_FLOAT: {
      return std::make_unique<TensorSummary<float>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements);
    }
    case DbgDataType::DT_FLOAT64: {
      return std::make_unique<TensorSummary<double>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements);
    }
    case DbgDataType::DT_BOOL: {
      return std::make_unique<TensorSummary<bool>>(tensor->GetDataPtr(), previous_tensor_ptr, num_elements);
    }
    default:
      MS_LOG(INFO) << "Unsupported tensor type";
      // return a null pointer
      return std::unique_ptr<TensorSummary<int32_t>>{};
  }
}

#ifdef OFFLINE_DBG_MODE
void *DebugServices::GetPrevTensor(const std::shared_ptr<TensorData> &tensor, bool previous_iter_tensor_needed) {
  void *previous_tensor_ptr = nullptr;
  std::shared_ptr<TensorData> tensor_prev;
  if (previous_iter_tensor_needed && tensor->GetIteration() > 1) {
    // read data in offline mode
    std::vector<std::string> file_paths;
    if (!is_sync_mode) {
      ConvertReadTensors(std::vector<std::string>{tensor->GetName()}, std::vector<size_t>{tensor->GetSlot()},
                         std::vector<unsigned int>{tensor->GetDeviceId()},
                         std::vector<unsigned int>{tensor->GetIteration() - 1},
                         std::vector<unsigned int>{tensor->GetRootGraphId()}, &file_paths);
    }
    std::vector<std::shared_ptr<TensorData>> result_list_prev;
    ReadDumpedTensor(std::vector<std::string>{tensor->GetName()}, std::vector<size_t>{tensor->GetSlot()},
                     std::vector<unsigned int>{tensor->GetDeviceId()},
                     std::vector<unsigned int>{tensor->GetIteration() - 1},
                     std::vector<unsigned int>{tensor->GetRootGraphId()}, file_paths, &result_list_prev);
    tensor_prev = result_list_prev[0];
    if (!tensor_prev->GetByteSize()) {
      tensor_prev.reset();
    } else {
      previous_tensor_ptr = tensor_prev->GetDataPtr();
    }
  }
  return previous_tensor_ptr;
}
#endif

void DebugServices::AddWatchPointsToCheck(bool init_dbg_suspend, bool step_end, bool recheck,
                                          const std::string &tensor_name, const std::string &tensor_name_no_slot,
                                          bool *previous_iter_tensor_needed, std::string *const qualified_tensor_name,
                                          std::vector<watchpoint_t> *const watchpoints_to_check) {
  for (auto w_table_item : watchpoint_table) {
    auto wp = std::get<1>(w_table_item);
    // check ONLY init conditions on initial suspended state.
    // skip other conditions on initial suspended state
    if (init_dbg_suspend && (wp.condition.type != INIT)) continue;
    // skip init condition if not init suspend
    if ((wp.condition.type == INIT) && !init_dbg_suspend) continue;
    // check change conditions only on step end.
    if (wp.change_condition() && !step_end) continue;
    // if recheck, ignore the cache results and reanalyze everything.
    // if not a recheck, check only unanalyzed tensors
    if (!recheck && wp_id_cache[tensor_name].count(wp.id)) continue;
    std::string found = wp.FindQualifiedTensorName(tensor_name_no_slot);
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
    wp_id_cache[tensor_name].insert(id);
  }
}

void DebugServices::CheckWatchpoints(std::vector<std::string> *const name, std::vector<std::string> *const slot,
                                     std::vector<int> *const condition, std::vector<unsigned int> *const watchpoint_id,
                                     std::vector<std::vector<parameter_t>> *const parameters,
                                     std::vector<int32_t> *const error_codes,
                                     const std::vector<std::string> &op_overflows,
                                     const std::vector<std::string> &async_file_pool,
                                     std::vector<std::shared_ptr<TensorData>> *tensor_list, const bool init_dbg_suspend,
                                     const bool step_end, const bool recheck, std::vector<unsigned int> *device_id,
                                     std::vector<unsigned int> *root_graph_id) {
  std::lock_guard<std::mutex> lg(lock_);
  if (watchpoint_table.empty()) return;
  // vector to store execution order of tensors hit
  std::vector<int> exec_order;
  for (auto &tensor : *tensor_list) {
#ifdef OFFLINE_DBG_MODE
    // read data in offline mode
    std::vector<std::shared_ptr<TensorData>> result_list;
    ReadDumpedTensor(std::vector<std::string>{tensor->GetName()}, std::vector<size_t>{tensor->GetSlot()},
                     std::vector<unsigned int>{tensor->GetDeviceId()},
                     std::vector<unsigned int>{tensor->GetIteration()},
                     std::vector<unsigned int>{tensor->GetRootGraphId()}, async_file_pool, &result_list);
    tensor = result_list[0];
    if (!tensor->GetByteSize()) {
      tensor.reset();
      continue;
    }
#endif
    const auto tensor_name = tensor->GetName();
    const auto tensor_name_no_slot = tensor_name.substr(0, tensor_name.find_first_of(':'));
    const auto tensor_slot = std::to_string(tensor->GetSlot());
    // no elements to analyze
    if (tensor->GetByteSize() == 0) continue;
    int tensor_dtype = tensor->GetType();
    std::vector<watchpoint_t> watchpoints_to_check;
    std::string qualified_tensor_name;
    bool previous_iter_tensor_needed = false;
    // Add do nothing line in case offline debug is off, prevent unused var warning
    (void)previous_iter_tensor_needed;
    AddWatchPointsToCheck(init_dbg_suspend, step_end, recheck, tensor_name, tensor_name_no_slot,
                          &previous_iter_tensor_needed, &qualified_tensor_name, &watchpoints_to_check);
    // no wp set on current tensor
    if (watchpoints_to_check.empty()) continue;
    uint32_t num_elements = tensor->GetNumElements();

#ifdef OFFLINE_DBG_MODE
    void *previous_tensor_ptr = GetPrevTensor(tensor, previous_iter_tensor_needed);
#else
    void *previous_tensor_ptr =
      tensor_loader_->GetPrevTensor(tensor_name) ? tensor_loader_->GetPrevTensor(tensor_name)->GetDataPtr() : nullptr;
#endif
    std::unique_ptr<ITensorSummary> base_summary_ptr;
    if (!(watchpoints_to_check.size() == 1 && watchpoints_to_check[0].condition.type == IS_OVERFLOW)) {
      base_summary_ptr = GetSummaryPtr(tensor, previous_tensor_ptr, num_elements, tensor_dtype);
      if (base_summary_ptr != nullptr) {
        base_summary_ptr->SummarizeTensor(watchpoints_to_check);
      }
    }
    for (auto &wp : watchpoints_to_check) {
      bool is_hit = false;
      int error_code = 0;
      std::vector<parameter_t> parameter_list = {};
      if (wp.condition.type == IS_OVERFLOW) {
        is_hit = (std::find(op_overflows.begin(), op_overflows.end(), tensor_name_no_slot) != op_overflows.end());
      } else if (base_summary_ptr != nullptr) {
        auto item = base_summary_ptr->IsWatchpointHit(wp);
        is_hit = std::get<ITensorSummary::eHitPos>(item);
        error_code = std::get<ITensorSummary::eErrorCodePos>(item);
        parameter_list = std::get<ITensorSummary::eParamListPos>(item);
      }
      AddAnalyzedTensorToCache(recheck, wp.id, tensor_name);
      if (is_hit || error_code) {
        std::vector<int>::iterator iter;
        // if the execution order is repeated, inserts the new one before the others with same execution order.
        iter = std::lower_bound(exec_order.begin(), exec_order.end(), tensor->GetExecutionOrder());
        int position = iter - exec_order.begin();
        exec_order.insert(iter, tensor->GetExecutionOrder());
        name->insert(name->begin() + position, qualified_tensor_name);
        slot->insert(slot->begin() + position, tensor_slot);
        condition->insert(condition->begin() + position, wp.condition.type);
        watchpoint_id->insert(watchpoint_id->begin() + position, wp.id);
        if (device_id != nullptr) {
          device_id->insert(device_id->begin() + position, tensor->GetDeviceId());
        }
        if (root_graph_id != nullptr) {
          root_graph_id->insert(root_graph_id->begin() + position, tensor->GetRootGraphId());
        }
        parameters->insert(parameters->begin() + position, parameter_list);
        error_codes->insert(error_codes->begin() + position, error_code);
      }
    }
#ifdef OFFLINE_DBG_MODE
    // in offline mode remove the need for the data
    tensor.reset();
#endif
  }
}

#ifdef OFFLINE_DBG_MODE
void DebugServices::ReadTensorFromNpy(const std::string &file_name, std::string *tensor_type, std::size_t *size,
                                      std::vector<int64_t> *shape, std::vector<char> **data_buffer) {
  std::ifstream infile;
  std::string file_path = file_name;
  MS_LOG(INFO) << "Reading in file: " << file_path;
  infile.open(file_path.c_str(), std::ios::ate | std::ios::binary | std::ios::in);
  if (!infile.is_open()) {
    MS_LOG(ERROR) << "Failed to open file (In ReadTensorFromNpy) " << file_path;
    return;
  }
  uint64_t file_size = infile.tellg();
  infile.seekg(0, std::ios::beg);
  std::unique_ptr<std::vector<char>> buffer(new std::vector<char>(file_size));
  if (!infile.read(buffer->data(), file_size)) {
    MS_LOG(ERROR) << "Failed to read file (In ReadTensorFromNpy) " << file_path;
    return;
  }
  uint16_t header_len = *reinterpret_cast<uint16_t *>(buffer->data() + 8);
  std::string header(buffer->data() + 9, header_len);
  std::size_t type_i = header.find("descr") + 10;
  *tensor_type = header.substr(type_i, 2);
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
  infile.seekg(header_len + 10);
  *data_buffer = new std::vector<char>(data_size);
  if (!infile.read((*data_buffer)->data(), data_size)) {
    MS_LOG(ERROR) << "Unable to get tensor data from npy";
  }
  *size = data_size;
}

void DebugServices::ConvertToHostFormat(const std::map<std::string, std::vector<std::string>> &dir_to_files_map,
                                        std::vector<std::string> *result_list) {
  std::string file_format = "npy";
  for (auto const &d : dir_to_files_map) {
    std::vector<std::string> files_to_convert_in_dir;
    std::string dump_key = d.first;
    for (auto const &file_name : d.second) {
      bool already_converted = false;
      for (std::string &file_found : *result_list) {
        if (file_found.find(file_name) != std::string::npos) {
          already_converted = true;
        }
      }
      if (!already_converted) {
        files_to_convert_in_dir.push_back(dump_key + "/" + file_name);
      }
    }
    std::ostringstream input_file_o;
    const char *const delim = " ";
    std::copy(files_to_convert_in_dir.begin(), files_to_convert_in_dir.end(),
              std::ostream_iterator<std::string>(input_file_o, delim));
    std::string input_files = input_file_o.str();
    MS_LOG(INFO) << "Ops to convert: " << input_files;
    if (input_files != "") {
      // Look for the installation path to the conver_async package. If not found, throw exception and terminate the
      // later task.
      try {
        auto pkg = pybind11::module::import("mindspore.offline_debug.convert_async");
        std::string convert_pkg_path = pkg.attr("__file__").cast<std::string>();
        MS_LOG(INFO) << "The file for converting async dump data is in " << convert_pkg_path;
        std::string convert_command = "python " + convert_pkg_path + " -out " + dump_key + " -t " + file_format +
                                      " -d " + dump_key + " -f NCHW -l " + input_files;
        (void)(system(convert_command.c_str()) + 1);
      } catch (pybind11::error_already_set &e) {
        MS_LOG(EXCEPTION) << "Can't find package mindspore.offline_debug.convert_async";
      }

      DIR *d_handle;
      d_handle = opendir(dump_key.c_str());
      if (d_handle != nullptr) {
        struct dirent *dir = nullptr;
        while ((dir = readdir(d_handle)) != NULL) {
          if (dir->d_type == DT_REG) {
            std::string candidate = dir->d_name;
            for (const std::string &file_to_find : files_to_convert_in_dir) {
              std::string file_n = file_to_find.substr(file_to_find.find_last_of("\\/") + 1);
              if (candidate.find(file_n) != std::string::npos && candidate.rfind(file_format) != std::string::npos) {
                // we found a converted file for this op
                result_list->push_back(dump_key + "/" + candidate);
              }
            }
          }
        }
      }
    }
  }
}

void DebugServices::ConvertReadTensors(std::vector<std::string> backend_name, std::vector<size_t> slot,
                                       std::vector<unsigned int> device_id, std::vector<unsigned int> iteration,
                                       std::vector<unsigned int> root_graph_id, std::vector<std::string> *result_list) {
  std::string file_format = "npy";
  std::map<std::string, std::vector<std::string>> dir_to_files_map;
  for (unsigned int i = 0; i < backend_name.size(); i++) {
    // form prefix of the tensor file to read from graph pb node name
    std::string dump_style_kernel_name = backend_name[i];
    const std::string strsrc = "/";

    std::string strdst = "_";

    std::string::size_type pos = 0;
    std::string::size_type srclen = strsrc.size();
    std::string::size_type dstlen = strdst.size();

    // remove slot from name
    std::size_t found_colon = dump_style_kernel_name.find_last_of(":");
    dump_style_kernel_name = dump_style_kernel_name.substr(0, found_colon);

    while ((pos = dump_style_kernel_name.find(strsrc, pos)) != std::string::npos) {
      dump_style_kernel_name.replace(pos, srclen, strdst);
      pos += dstlen;
    }

    std::string prefix_dump_file_name = dump_style_kernel_name;

    std::string specific_dump_dir = dump_dir + "/rank_" + std::to_string(device_id[i]) + "/" + net_name + "/" +
                                    std::to_string(root_graph_id[i]) + "/" + std::to_string(iteration[i]);

    // search files in dir for the one that meets the filename prefix and read the file into memory
    DIR *d;
    d = opendir(specific_dump_dir.c_str());
    if (d != nullptr) {
      struct dirent *dir = nullptr;
      while ((dir = readdir(d)) != NULL) {
        if (dir->d_type == DT_REG) {
          std::string file_name = dir->d_name;
          std::string file_name_w_o_perfix = file_name.substr(file_name.find('.') + 1);
          if (file_name_w_o_perfix.rfind(prefix_dump_file_name, 0) == 0 &&
              file_name.rfind(file_format) == std::string::npos) {
            // if file matches prefix and is in device format add to candidate files to convert.
            dir_to_files_map[specific_dump_dir].push_back(file_name);
          } else if (file_name_w_o_perfix.rfind(prefix_dump_file_name, 0) == 0 &&
                     file_name.rfind(file_format) != std::string::npos) {
            // otherwise, if file matches prefix and already has been converted to host format
            // add to result of converted files.
            result_list->push_back(specific_dump_dir + "/" + file_name);
          }
        }
      }
    }
    closedir(d);
  }
  ConvertToHostFormat(dir_to_files_map, result_list);
}

void DebugServices::ConvertWatchPointNodes(const std::vector<std::tuple<std::string, std::string>> &proto_dump,
                                           const std::string &specific_dump_dir,
                                           std::vector<std::string> *result_list) {
  std::string file_format = "npy";
  std::map<std::string, std::vector<std::string>> dir_to_files_map;
  for (const auto &node : proto_dump) {
    std::string dump_name = std::get<1>(node);
    // search files in dir for the one that meets the filename prefix and read the file into memory
    DIR *d;
    d = opendir(specific_dump_dir.c_str());
    if (d != nullptr) {
      struct dirent *dir = nullptr;
      while ((dir = readdir(d)) != NULL) {
        if (dir->d_type == DT_REG) {
          std::string file_name = dir->d_name;
          std::string file_name_w_o_perfix = file_name.substr(file_name.find('.') + 1);
          if (file_name_w_o_perfix.rfind(dump_name, 0) == 0 && file_name.rfind(file_format) == std::string::npos) {
            // if file matches prefix and is in device format add to candidate files to convert.
            dir_to_files_map[specific_dump_dir].push_back(file_name);
          } else if (file_name_w_o_perfix.rfind(dump_name, 0) == 0 &&
                     file_name.rfind(file_format) != std::string::npos) {
            // otherwise, if file matches prefix and already has been converted to host format
            // add to result of converted files.
            result_list->push_back(specific_dump_dir + "/" + file_name);
          }
        }
      }
    }
    closedir(d);
  }
  ConvertToHostFormat(dir_to_files_map, result_list);
}

void DebugServices::GetTensorDataInfoAsync(const std::vector<std::tuple<std::string, std::string>> &proto_dump,
                                           uint32_t iteration, uint32_t device_id, uint32_t root_graph_id,
                                           const std::vector<std::string> &async_file_pool,
                                           std::vector<std::shared_ptr<TensorData>> *tensor_list) {
  for (auto &node : proto_dump) {
    std::vector<size_t> slot_list;
    for (const std::string &file_name : async_file_pool) {
      std::string dump_name = std::get<1>(node);
      std::size_t found = file_name.find(dump_name);
      std::size_t found_out = file_name.find("output");
      std::size_t found_dot_start = file_name.find(".", found_out);
      std::size_t found_dot_end = file_name.find(".", found_dot_start);

      if (found != std::string::npos && found_out != std::string::npos) {
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
      tensor_data->SetDataPtr(NULL);
      tensor_data->SetByteSize(0);
      tensor_data->SetType("");
      tensor_data->SetShape(shape);

      tensor_list->push_back(tensor_data);
    }
  }
}

void DebugServices::AddToTensorData(const std::string &backend_name, const std::size_t slot,
                                    const unsigned int iteration, const unsigned int device_id,
                                    const unsigned int root_graph_id, const std::size_t data_size,
                                    const std::string &type_name, const std::vector<int64_t> &shape,
                                    std::vector<char> *buffer, std::vector<std::shared_ptr<TensorData>> *result_list) {
  // call LoadNewTensor to store tensor in internal cache
  auto tensor_data = std::make_shared<TensorData>();
  tensor_data->SetName(backend_name);
  tensor_data->SetExecutionOrder(0);
  tensor_data->SetSlot(slot);
  tensor_data->SetIteration(iteration);
  tensor_data->SetDeviceId(device_id);
  tensor_data->SetRootGraphId(root_graph_id);
  if (data_size) {
    tensor_data->SetDataPtr(buffer->data());
  } else {
    tensor_data->SetDataPtr(NULL);
  }
  tensor_data->SetByteSize(data_size);
  tensor_data->SetType(type_name);
  tensor_data->SetShape(shape);
  if (data_size) {
    tensor_loader_->LoadNewTensor(tensor_data, false);
  }

  // add to result_list
  result_list->push_back(tensor_data);
}

void DebugServices::ReadDumpedTensor(std::vector<std::string> backend_name, std::vector<size_t> slot,
                                     std::vector<unsigned int> device_id, std::vector<unsigned int> iteration,
                                     std::vector<unsigned int> root_graph_id,
                                     const std::vector<std::string> &async_file_pool,
                                     std::vector<std::shared_ptr<TensorData>> *result_list) {
  for (unsigned int i = 0; i < backend_name.size(); i++) {
    // form prefix of the tensor file to read from graph pb node name
    std::string dump_style_kernel_name = backend_name[i];
    const std::string strsrc = "/";

    std::string strdst;
    if (is_sync_mode) {
      strdst = "--";
    } else {
      strdst = "_";
    }

    std::string::size_type pos = 0;
    std::string::size_type srclen = strsrc.size();
    std::string::size_type dstlen = strdst.size();

    // remove slot from name
    std::size_t found_colon = dump_style_kernel_name.find_last_of(":");
    dump_style_kernel_name = dump_style_kernel_name.substr(0, found_colon);

    while ((pos = dump_style_kernel_name.find(strsrc, pos)) != std::string::npos) {
      dump_style_kernel_name.replace(pos, srclen, strdst);
      pos += dstlen;
    }

    std::string prefix_dump_file_name;
    if (is_sync_mode) {
      prefix_dump_file_name = dump_style_kernel_name.substr(dump_style_kernel_name.rfind("--") + 2);
      prefix_dump_file_name += ".output." + std::to_string(slot[i]);
    } else {
      prefix_dump_file_name = dump_style_kernel_name;
    }

    std::string specific_dump_dir = dump_dir + "/rank_" + std::to_string(device_id[i]) + "/" + net_name + "/" +
                                    std::to_string(root_graph_id[i]) + "/" + std::to_string(iteration[i]);

    // search files in dir for the one that meets the filename prefix and read the file into memory
    std::vector<char> *buffer = NULL;
    std::string type_name = "";
    std::vector<int64_t> shape;
    uint64_t data_size = 0;
    if (is_sync_mode) {
      DIR *d;
      d = opendir(specific_dump_dir.c_str());
      if (d != nullptr) {
        struct dirent *dir = nullptr;
        bool found_file = false;
        while ((dir = readdir(d)) != NULL) {
          if (dir->d_type == DT_REG) {
            std::string file_name = dir->d_name;

            // strip off the task_id, stream_id, and timestamp, then compare
            size_t first_dot = file_name.find(".");
            size_t second_dot = file_name.find(".", first_dot + 1);
            size_t seventh_dot = file_name.rfind(".", file_name.rfind(".") - 1);
            size_t fifth_dot = file_name.rfind(".", file_name.rfind(".", seventh_dot - 1) - 1);
            if (fifth_dot == std::string::npos) {
              continue;
            }
            std::string start_string = file_name.substr(first_dot + 1, second_dot - first_dot - 1);
            std::string end_string = file_name.substr(fifth_dot, seventh_dot - fifth_dot);
            std::string stripped_file_name = start_string + end_string;
            std::size_t found = stripped_file_name.rfind(prefix_dump_file_name, 0);

            if (found != 0) {
              continue;
            }

            shape.clear();
            std::string full_path = specific_dump_dir + "/" + file_name;
            ReadTensorFromNpy(full_path, &type_name, &data_size, &shape, &buffer);
            AddToTensorData(backend_name[i], slot[i], iteration[i], device_id[i], root_graph_id[i], data_size,
                            type_name, shape, buffer, result_list);
            found_file = true;
          }
        }
        if (!found_file) {
          AddToTensorData(backend_name[i], slot[i], iteration[i], device_id[i], root_graph_id[i], 0, type_name, shape,
                          buffer, result_list);
        }
      } else {
        MS_LOG(INFO) << "directory does not exist!";
      }
      closedir(d);
    } else {
      bool found = false;
      // if async mode
      for (const std::string &file_path : async_file_pool) {
        if (file_path.find(prefix_dump_file_name) != std::string::npos &&
            file_path.find(".output." + std::to_string(slot[i])) != std::string::npos) {
          found = true;
          shape.clear();
          ReadTensorFromNpy(file_path, &type_name, &data_size, &shape, &buffer);
          AddToTensorData(backend_name[i], slot[i], iteration[i], device_id[i], root_graph_id[i], data_size, type_name,
                          shape, buffer, result_list);
        }
      }
      // If no npy file is found, add empty tensor data.
      if (!found) {
        AddToTensorData(backend_name[i], slot[i], iteration[i], device_id[i], root_graph_id[i], 0, type_name, shape,
                        buffer, result_list);
      }
    }
  }
}

void ReplaceSrcFileName(const bool is_sync_mode, std::string *dump_style_name) {
  if (dump_style_name == nullptr) {
    return;
  }
  const std::string strsrc = "/";
  std::string strdst;
  if (is_sync_mode) {
    strdst = "--";
  } else {
    strdst = "_";
  }
  std::string::size_type pos = 0;
  std::string::size_type srclen = strsrc.size();
  std::string::size_type dstlen = strdst.size();

  while ((pos = dump_style_name->find(strsrc, pos)) != std::string::npos) {
    dump_style_name->replace(pos, srclen, strdst);
    pos += dstlen;
  }
}

std::vector<std::shared_ptr<TensorData>> DebugServices::ReadNeededDumpedTensors(
  unsigned int iteration, std::vector<std::string> *async_file_pool) {
  // get a list of nodes and the devices they are on to monitor
  std::vector<std::shared_ptr<TensorData>> tensor_list;
  std::map<std::tuple<uint32_t, uint32_t>, std::unordered_set<std::string>> device_and_graph_to_nodes;
  for (auto w_table_item : watchpoint_table) {
    auto wp = std::get<1>(w_table_item);
    for (auto check_node : wp.check_node_list) {
      unsigned int index = 0;
      std::string w_name = std::get<0>(check_node);
      bool w_is_param = std::get<1>(check_node);

      std::string node_name = w_name;
      if (w_is_param) {
        std::size_t found = node_name.find_last_of("/");
        node_name = node_name.substr(found + 1);
      }

      std::vector<uint32_t> devices = std::get<1>(wp.check_node_device_list[index]);
      std::vector<uint32_t> graphs = std::get<1>(wp.check_node_graph_list[index]);
      for (auto device : devices) {
        for (auto graph : graphs) {
          std::tuple<uint32_t, uint32_t> key(device, graph);
          device_and_graph_to_nodes[key].insert(node_name);
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
    std::unordered_set<std::string> wp_nodes = device_and_graph_item.second;
    std::vector<std::tuple<std::string, std::string>> proto_to_dump;

    std::string specific_dump_dir = dump_dir + "/rank_" + std::to_string(device_id) + "/" + net_name + "/" +
                                    std::to_string(root_graph_id) + "/" + std::to_string(iteration);

    // convert node names to dump style
    for (auto node : wp_nodes) {
      std::string orig_name = node;
      std::string dump_style_name = node;
      ReplaceSrcFileName(is_sync_mode, &dump_style_name);

      if (is_sync_mode) {
        dump_style_name = dump_style_name.substr(dump_style_name.rfind("--") + 2);
        dump_style_name.append(".output.");
      }

      proto_to_dump.push_back(std::tuple<std::string, std::string>(orig_name, dump_style_name));
    }

    if (!is_sync_mode) {
      // convert all files in proto_to_dump to npy and add to pool of async file names
      ConvertWatchPointNodes(proto_to_dump, specific_dump_dir, async_file_pool);
    }
    if (is_sync_mode) {
      // search files in dir for the one that meets the filename prefix and read the file into memory
      DIR *d;
      d = opendir(specific_dump_dir.c_str());
      if (d != nullptr) {
        struct dirent *dir = nullptr;
        while ((dir = readdir(d)) != NULL) {
          if (dir->d_type == DT_REG) {
            std::string file_name = dir->d_name;
            for (auto &node : proto_to_dump) {
              std::string dump_name = std::get<1>(node);

              // strip off the task_id, stream_id, and timestamp, then compare
              size_t first_dot = file_name.find(".");
              size_t second_dot = file_name.find(".", first_dot + 1);
              size_t seventh_dot = file_name.rfind(".", file_name.rfind(".") - 1);
              size_t fifth_dot = file_name.rfind(".", file_name.rfind(".", seventh_dot - 1) - 1);
              if (fifth_dot == std::string::npos) {
                continue;
              }

              std::string start_string = file_name.substr(first_dot + 1, second_dot - first_dot - 1);
              std::string end_string = file_name.substr(fifth_dot, seventh_dot - fifth_dot);
              std::string stripped_file_name = start_string + end_string;
              std::size_t found = stripped_file_name.rfind(dump_name, 0);

              if (found == 0) {
                size_t slot = std::stoul(stripped_file_name.substr(dump_name.length()));
                std::vector<int64_t> shape;
                std::string orig_name = std::get<0>(node);
                AddToTensorData(orig_name, slot, iteration, device_id, root_graph_id, 0, "", shape, NULL, &tensor_list);
                break;
              }
            }
          }
        }
      }
    } else {
      GetTensorDataInfoAsync(proto_to_dump, iteration, device_id, root_graph_id, *async_file_pool, &tensor_list);
    }
  }

  return tensor_list;
}
#endif

void DebugServices::ReadNodesTensors(const std::vector<std::string> &name, std::vector<std::string> *const ret_name,
                                     std::vector<char *> *const data_ptr, std::vector<ssize_t> *const data_size,
                                     std::vector<unsigned int> *const dtype,
                                     std::vector<std::vector<int64_t>> *const shape) {
  std::vector<std::tuple<std::string, std::shared_ptr<TensorData>>> result_list;
  tensor_loader_->SearchTensors(name, &result_list);

  for (auto result : result_list) {
    if (!std::get<1>(result)) {
      continue;
    }
    ret_name->push_back(std::get<0>(result));
    data_ptr->push_back(reinterpret_cast<char *>(std::get<1>(result)->GetDataPtr()));
    data_size->push_back(std::get<1>(result)->GetByteSize());
    dtype->push_back(std::get<1>(result)->GetType());
    shape->push_back(std::get<1>(result)->GetShape());
  }
}

#ifdef ONLINE_DBG_MODE
bool DebugServices::IsWatchPoint(const std::string &kernel_name, const CNodePtr &kernel) const {
  bool ret = false;
  for (auto w_table_item : watchpoint_table) {
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
  if (kernel) {
    auto input_size = AnfAlgo::GetInputTensorNum(kernel);
    for (size_t j = 0; j < input_size; ++j) {
      auto input_kernel = kernel->input(j + 1);
      std::string input_kernel_name = input_kernel->fullname_with_scope();
      auto found = w_name.find_last_of('/');
      if (found != std::string::npos && w_name.substr(found + 1) == input_kernel_name) return true;
    }
    return false;
  } else {
    return false;
  }
}
#endif

void DebugServices::EmptyTensor() { tensor_loader_->EmptyTensor(); }

std::vector<std::shared_ptr<TensorData>> DebugServices::GetTensor() const { return tensor_loader_->GetTensor(); }

std::vector<std::shared_ptr<TensorData>> DebugServices::GetNodeTensorMap(const std::string &node_name) const {
  return tensor_loader_->GetNodeTensorMap(node_name);
}

uint32_t DebugServices::GetTensorLoaderIterNum() const { return tensor_loader_->GetIterNum(); }

void DebugServices::SetTensorLoaderIterNum(uint32_t iter_num) { tensor_loader_->set_iter_num(iter_num); }

void DebugServices::EmptyPrevTensor() { tensor_loader_->EmptyPrevTensor(); }

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

std::unordered_map<unsigned int, DebugServices::watchpoint_t> DebugServices::GetWatchpointTable() {
  return watchpoint_table;
}

void DebugServices::ResetLoadedTensors() {
  wp_id_cache.clear();
  MS_LOG(INFO) << "Resetting loaded tensors";
  tensor_loader_->MoveParametersCurrentToPrev();
  tensor_loader_->EmptyCurrentTensor();
  // will move parameters from previous to current map
  tensor_loader_->SwapCurrentPrev();
}

#ifdef ONLINE_DBG_MODE
std::vector<std::shared_ptr<TensorData>> DebugServices::GetNodeTensor(const CNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  std::vector<std::shared_ptr<TensorData>> result;
  auto output_size = AnfAlgo::GetOutputTensorNum(kernel);
  auto kernel_name = kernel->fullname_with_scope();
  for (size_t j = 0; j < output_size; ++j) {
    auto tensor_name_with_slot = kernel_name + ":" + std::to_string(j);
    auto tensor = tensor_loader_->GetTensor(tensor_name_with_slot);
    if (tensor) result.push_back(tensor);
  }
  return result;
}
#endif

bool DebugServices::TensorExistsInCurrent(const std::string &tensor_name) {
  return tensor_loader_->TensorExistsInCurrent(tensor_name);
}
void DebugServices::MoveTensorCurrentToPrev(const std::string &tensor_name) {
  tensor_loader_->MoveTensorCurrentToPrev(tensor_name);
}

void DebugServices::SetNetName(std::string net_name) { this->net_name = net_name; }

std::string DebugServices::GetNetName() { return net_name; }

void DebugServices::SetDumpDir(std::string dump_dir) { this->dump_dir = dump_dir; }

std::string DebugServices::GetDumpDir() { return dump_dir; }

void DebugServices::SetSyncMode(bool is_sync_mode) { this->is_sync_mode = is_sync_mode; }

bool DebugServices::GetSyncMode() { return is_sync_mode; }

#ifdef ONLINE_DBG_MODE
}  // namespace mindspore
#endif
