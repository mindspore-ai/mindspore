/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include <map>
#include "backend/session/anf_runtime_algorithm.h"
#include "debug/debug_services.h"
#include "debug/debugger/tensor_summary.h"

namespace mindspore {

DebugServices::DebugServices() {
  tensor_loader_ = new TensorLoader();
  uint32_t iter_num = -1;
  tensor_loader_->set_iter_num(iter_num);
}

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

DebugServices::~DebugServices() { delete tensor_loader_; }

void DebugServices::AddWatchpoint(unsigned int id, unsigned int watch_condition, float parameter,
                                  const std::vector<std::tuple<std::string, bool>> &check_node_list,
                                  const std::vector<parameter_t> &parameter_list) {
  std::lock_guard<std::mutex> lg(lock_);

  watchpoint_t watchpoint_item;
  watchpoint_item.id = id;
  watchpoint_item.condition.type = static_cast<CONDITION_TYPE>(watch_condition);
  watchpoint_item.condition.parameter = parameter;
  watchpoint_item.check_node_list = check_node_list;
  watchpoint_item.parameter_list = parameter_list;
  watchpoint_table[id] = watchpoint_item;
}

void DebugServices::RemoveWatchpoint(unsigned int id) {
  std::lock_guard<std::mutex> lg(lock_);
  watchpoint_table.erase(id);
}

std::unique_ptr<ITensorSummary> GetSummaryPtr(const mindspore::tensor::TensorPtr &tensor_ptr, void *previous_tensor_ptr,
                                              uint32_t num_elements, int tensor_dtype) {
  switch (tensor_dtype) {
    case kNumberTypeUInt8: {
      return std::make_unique<TensorSummary<uint8_t>>(tensor_ptr->data_c(), previous_tensor_ptr, num_elements);
    }
    case kNumberTypeInt8: {
      return std::make_unique<TensorSummary<int8_t>>(tensor_ptr->data_c(), previous_tensor_ptr, num_elements);
    }
    case kNumberTypeUInt16: {
      return std::make_unique<TensorSummary<uint16_t>>(tensor_ptr->data_c(), previous_tensor_ptr, num_elements);
    }
    case kNumberTypeInt16: {
      return std::make_unique<TensorSummary<int16_t>>(tensor_ptr->data_c(), previous_tensor_ptr, num_elements);
    }
    case kNumberTypeUInt32: {
      return std::make_unique<TensorSummary<uint32_t>>(tensor_ptr->data_c(), previous_tensor_ptr, num_elements);
    }
    case kNumberTypeInt32:
    case kNumberTypeInt: {
      return std::make_unique<TensorSummary<int32_t>>(tensor_ptr->data_c(), previous_tensor_ptr, num_elements);
    }
    case kNumberTypeUInt64: {
      return std::make_unique<TensorSummary<uint64_t>>(tensor_ptr->data_c(), previous_tensor_ptr, num_elements);
    }
    case kNumberTypeInt64: {
      return std::make_unique<TensorSummary<int64_t>>(tensor_ptr->data_c(), previous_tensor_ptr, num_elements);
    }
    case kNumberTypeFloat16: {
      return std::make_unique<TensorSummary<float16>>(tensor_ptr->data_c(), previous_tensor_ptr, num_elements);
    }
    case kNumberTypeFloat32:
    case kNumberTypeFloat: {
      return std::make_unique<TensorSummary<float>>(tensor_ptr->data_c(), previous_tensor_ptr, num_elements);
    }
    case kNumberTypeFloat64: {
      return std::make_unique<TensorSummary<double>>(tensor_ptr->data_c(), previous_tensor_ptr, num_elements);
    }
    case kNumberTypeBool: {
      return std::make_unique<TensorSummary<bool>>(tensor_ptr->data_c(), previous_tensor_ptr, num_elements);
    }
    default:
      MS_LOG(INFO) << "Unsupported tensor type";
      // return a null pointer
      return std::unique_ptr<TensorSummary<bool>>(nullptr);
  }
}

void DebugServices::AddWatchPointsToCheck(bool init_dbg_suspend, bool step_end, bool recheck,
                                          const std::string &tensor_name, const std::string &tensor_name_no_slot,
                                          std::string *qualified_tensor_name,
                                          std::vector<watchpoint_t> *watchpoints_to_check) {
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
    }
  }
}

void DebugServices::CheckWatchpoints(std::vector<std::string> *name, std::vector<std::string> *slot,
                                     std::vector<int> *condition, std::vector<unsigned int> *const watchpoint_id,
                                     std::vector<std::vector<parameter_t>> *parameters,
                                     std::vector<int32_t> *error_codes, const std::vector<std::string> &op_overflows,
                                     const std::vector<std::shared_ptr<TensorData>> &tensor_list,
                                     const bool init_dbg_suspend, const bool step_end, const bool recheck) {
  std::lock_guard<std::mutex> lg(lock_);
  if (watchpoint_table.empty()) return;

  for (const auto &tensor : tensor_list) {
    const auto tensor_name = tensor->GetName();
    const auto tensor_name_no_slot = tensor_name.substr(0, tensor_name.find_first_of(':'));
    const auto tensor_slot = std::to_string(tensor->GetSlot());
    mindspore::tensor::TensorPtr tensor_ptr = tensor->GetTensor();
    // no elements to analyze
    if (tensor_ptr->DataSize() == 0) continue;
    int tensor_dtype = tensor_ptr->data_type_c();
    std::vector<watchpoint_t> watchpoints_to_check;
    std::string qualified_tensor_name;
    AddWatchPointsToCheck(init_dbg_suspend, step_end, recheck, tensor_name, tensor_name_no_slot, &qualified_tensor_name,
                          &watchpoints_to_check);
    // no wp set on current tensor
    if (watchpoints_to_check.empty()) continue;

    uint32_t num_elements = tensor_ptr->DataSize();
    void *previous_tensor_ptr = tensor_loader_->GetPrevTensor(tensor_name)
                                  ? tensor_loader_->GetPrevTensor(tensor_name)->GetTensor()->data_c()
                                  : nullptr;
    std::unique_ptr<ITensorSummary> base_summary_ptr;
    if (!(watchpoints_to_check.size() == 1 && watchpoints_to_check[0].condition.type == IS_OVERFLOW)) {
      base_summary_ptr = GetSummaryPtr(tensor_ptr, previous_tensor_ptr, num_elements, tensor_dtype);
      if (base_summary_ptr != nullptr) {
        base_summary_ptr->SummarizeTensor(watchpoints_to_check);
      } else {
        continue;
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
        is_hit = std::get<0>(item);
        error_code = std::get<1>(item);
        parameter_list = std::get<2>(item);
      }
      // add analyzed tensor to cache
      if (!recheck) {
        wp_id_cache[tensor_name].insert(wp.id);
      }

      if (is_hit || error_code) {
        name->push_back(qualified_tensor_name);
        slot->push_back(tensor_slot);
        condition->push_back(wp.condition.type);
        watchpoint_id->push_back(wp.id);
        parameters->push_back(parameter_list);
        error_codes->push_back(error_code);
      }
    }
  }
}

void DebugServices::ReadNodesTensors(std::vector<std::string> name, std::vector<std::string> *ret_name,
                                     std::vector<char *> *data_ptr, std::vector<ssize_t> *data_size,
                                     std::vector<TypePtr> *dtype, std::vector<std::vector<int64_t>> *const shape) {
  std::vector<std::tuple<std::string, std::shared_ptr<TensorData>>> result_list;
  tensor_loader_->SearchTensors(name, &result_list);

  for (auto result : result_list) {
    if (!std::get<1>(result)) {
      continue;
    }
    ret_name->push_back(std::get<0>(result));
    data_ptr->push_back(reinterpret_cast<char *>(std::get<1>(result)->GetTensor()->data_c()));
    data_size->push_back(std::get<1>(result)->GetTensor()->data().nbytes());
    dtype->push_back(std::get<1>(result)->GetTensor()->Dtype());
    shape->push_back(std::get<1>(result)->GetTensor()->shape());
  }
}

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
      if (found != std::string::npos && w_name.size() > found + 1 && w_name.substr(found + 1) == input_kernel_name)
        return true;
    }
    return false;
  } else {
    return false;
  }
}

void DebugServices::EmptyTensor() { tensor_loader_->EmptyTensor(); }

std::vector<std::shared_ptr<TensorData>> DebugServices::GetTensor() const { return tensor_loader_->GetTensor(); }

std::vector<std::shared_ptr<TensorData>> DebugServices::GetNodeTensorMap(const std::string &node_name) const {
  return tensor_loader_->GetNodeTensorMap(node_name);
}

uint32_t DebugServices::GetTensorLoaderIterNum() const { return tensor_loader_->GetIterNum(); }

void DebugServices::SetTensorLoaderIterNum(uint32_t iter_num) { tensor_loader_->set_iter_num(iter_num); }

void DebugServices::EmptyPrevTensor() { tensor_loader_->EmptyPrevTensor(); }

void DebugServices::EmptyCurrentTensor() { tensor_loader_->EmptyCurrentTensor(); }

bool DebugServices::DumpTensorToFile(const std::string &tensor_name, bool trans_flag, const std::string &filepath,
                                     const std::string &host_fmt, const std::vector<int64_t> &host_shape,
                                     TypeId host_type, TypeId addr_type_id, const std::string &addr_format,
                                     size_t slot) const {
  return tensor_loader_->DumpTensorToFile(tensor_name, trans_flag, filepath, host_fmt, host_shape, host_type,
                                          addr_type_id, addr_format, slot);
}

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
bool DebugServices::TensorExistsInCurrent(std::string tensor_name) {
  return tensor_loader_->TensorExistsInCurrent(tensor_name);
}
void DebugServices::MoveTensorCurrentToPrev(std::string tensor_name) {
  tensor_loader_->MoveTensorCurrentToPrev(tensor_name);
}

}  // namespace mindspore
