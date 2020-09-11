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
#include "debug/debug_services.h"
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
                                  const std::vector<std::tuple<std::string, bool>> &check_node_list) {
  std::lock_guard<std::mutex> lg(lock_);

  watchpoint_t watchpoint_item;
  watchpoint_item.id = id;
  watchpoint_item.condition.type = static_cast<CONDITION_TYPE>(watch_condition);
  watchpoint_item.condition.parameter = parameter;
  if (watch_condition > 2)
    // odd indices are greater than conditions and even indicies are less than
    watchpoint_item.condition.comparison = (watch_condition & 1) == 0 ? "LT" : "GT";
  watchpoint_item.check_node_list = check_node_list;
  watchpoint_table[id] = watchpoint_item;
}

void DebugServices::RemoveWatchpoint(unsigned int id) {
  std::lock_guard<std::mutex> lg(lock_);
  watchpoint_table.erase(id);
}

template <typename T>
DebugServices::tensor_stats DebugServices::SummarizeTensor(const T *start, unsigned int n, bool need_min_max,
                                                           bool need_mean_sd) {
  tensor_stats stats;
  for (unsigned int i = 0; i < n; ++i) {
    auto val = static_cast<double>(start[i]);
    stats.has_nan = stats.has_nan || std::isnan(val);
    stats.has_inf = stats.has_inf || std::isinf(val);
    if (stats.has_inf && stats.has_nan) {
      // other statistics don't make sense in this case
      break;
    }

    if (need_min_max) {
      stats.min = std::min(stats.min, val);
      stats.max = std::max(stats.max, val);
    }

    if (need_mean_sd) {
      double delta = val - stats.mean;
      stats.mean += delta / (i + 1);
      stats.m2 += delta * (val - stats.mean);
    }
  }
  stats.n = n;
  return stats;
}

void DebugServices::CheckWatchpoints(std::vector<std::string> *name, std::vector<std::string> *slot,
                                     std::vector<int> *condition, std::vector<unsigned int> *watchpoint_id,
                                     const std::vector<std::string> &op_overflows,
                                     const std::vector<std::shared_ptr<TensorData>> &tensor_list) {
  std::lock_guard<std::mutex> lg(lock_);
  if (watchpoint_table.empty()) {
    return;
  }

  for (const auto &tensor : tensor_list) {
    const auto tensor_name = tensor->GetName();
    const auto tensor_name_no_slot = tensor_name.substr(0, tensor_name.find_first_of(':'));
    const auto tensor_slot = std::to_string(tensor->GetSlot());
    mindspore::tensor::TensorPtr tensor_ptr = tensor->GetTensor();
    int tensor_dtype = tensor_ptr->data_type_c();
    std::vector<unsigned int> hit_encountered;
    std::unordered_map<unsigned int, watchpoint_t> watchpoints_to_check_table;
    bool min_max_enabled = false;
    bool mean_sd_enabled = false;
    bool inf_nan_enabled = false;
    for (auto w_table_item : watchpoint_table) {
      auto wp = std::get<1>(w_table_item);
      if (wp.condition.type != IS_OVERFLOW && tensor_dtype == kNumberTypeBool) continue;
      if (wp.IsNodeIncluded(tensor_name_no_slot)) {
        min_max_enabled |= wp.min_max_enabled();
        mean_sd_enabled |= wp.mean_sd_enabled();
        inf_nan_enabled |= wp.inf_nan_enabled();
        watchpoints_to_check_table[w_table_item.second.id] = w_table_item.second;
      }
    }
    tensor_stats stats;
    uint num_elements = tensor_ptr->DataSize();
    if (min_max_enabled || mean_sd_enabled || inf_nan_enabled) {
      switch (tensor_dtype) {
        case kNumberTypeUInt8: {
          auto start_addr = reinterpret_cast<uint8_t *>(tensor_ptr->data_c());
          stats = SummarizeTensor(start_addr, num_elements, min_max_enabled, mean_sd_enabled);
          break;
        }
        case kNumberTypeInt8: {
          auto start_addr = reinterpret_cast<int8_t *>(tensor_ptr->data_c());
          stats = SummarizeTensor(start_addr, num_elements, min_max_enabled, mean_sd_enabled);
          break;
        }
        case kNumberTypeUInt16: {
          auto start_addr = reinterpret_cast<uint16_t *>(tensor_ptr->data_c());
          stats = SummarizeTensor(start_addr, num_elements, min_max_enabled, mean_sd_enabled);
          break;
        }
        case kNumberTypeInt16: {
          auto start_addr = reinterpret_cast<int16_t *>(tensor_ptr->data_c());
          stats = SummarizeTensor(start_addr, num_elements, min_max_enabled, mean_sd_enabled);
          break;
        }
        case kNumberTypeUInt32: {
          auto start_addr = reinterpret_cast<uint32_t *>(tensor_ptr->data_c());
          stats = SummarizeTensor(start_addr, num_elements, min_max_enabled, mean_sd_enabled);
          break;
        }
        case kNumberTypeInt32:
        case kNumberTypeInt: {
          auto start_addr = reinterpret_cast<int32_t *>(tensor_ptr->data_c());
          stats = SummarizeTensor(start_addr, num_elements, min_max_enabled, mean_sd_enabled);
          break;
        }
        case kNumberTypeUInt64: {
          auto start_addr = reinterpret_cast<uint64_t *>(tensor_ptr->data_c());
          stats = SummarizeTensor(start_addr, num_elements, min_max_enabled, mean_sd_enabled);
          break;
        }
        case kNumberTypeInt64: {
          auto start_addr = reinterpret_cast<int64_t *>(tensor_ptr->data_c());
          stats = SummarizeTensor(start_addr, num_elements, min_max_enabled, mean_sd_enabled);
          break;
        }
        case kNumberTypeFloat16: {
          auto start_addr = reinterpret_cast<float16 *>(tensor_ptr->data_c());
          stats = SummarizeTensor(start_addr, num_elements, min_max_enabled, mean_sd_enabled);
          break;
        }
        case kNumberTypeFloat32:
        case kNumberTypeFloat: {
          auto start_addr = reinterpret_cast<float *>(tensor_ptr->data_c());
          stats = SummarizeTensor(start_addr, num_elements, min_max_enabled, mean_sd_enabled);
          break;
        }
        case kNumberTypeFloat64: {
          auto start_addr = reinterpret_cast<double *>(tensor_ptr->data_c());
          stats = SummarizeTensor(start_addr, num_elements, min_max_enabled, mean_sd_enabled);
          break;
        }
        default:
          MS_LOG(INFO) << "Unsupported tensor type";
          break;
      }
    }

    for (auto &it : watchpoints_to_check_table) {
      auto wp_id = it.second.id;
      CONDITION_TYPE enabled_condition = it.second.condition.type;
      bool hit = (enabled_condition == HAS_NAN && stats.has_nan) || (enabled_condition == HAS_INF && stats.has_inf) ||
                 (enabled_condition == IS_OVERFLOW &&
                  std::find(op_overflows.begin(), op_overflows.end(), tensor_name_no_slot) != op_overflows.end());

      if (enabled_condition > 2) {
        if (stats.has_inf || stats.has_nan) {
          MS_LOG(WARNING) << "NaN or/and INF present in tensor: " << tensor_name << ". Cannot check "
                          << condition_label[enabled_condition] << " watchpoint.";
        } else {
          bool gt = stats.statLookup(enabled_condition) > it.second.condition.parameter;
          bool lt = stats.statLookup(enabled_condition) < it.second.condition.parameter;
          hit |= it.second.condition.comparison == "GT" ? gt : lt;
        }
      }
      if (hit) hit_encountered.push_back(wp_id);
    }

    for (auto it_hit_id = hit_encountered.begin(); it_hit_id != hit_encountered.end(); ++it_hit_id) {
      if (watchpoint_table.find(*it_hit_id) != watchpoint_table.end()) {
        name->push_back(tensor_name_no_slot);
        slot->push_back(tensor_slot);
        int condition_item = watchpoint_table.find(*it_hit_id)->second.condition.type;
        condition->push_back(condition_item);
        watchpoint_id->push_back(*it_hit_id);
      }
      watchpoints_to_check_table.erase(*it_hit_id);
    }
  }
}

void DebugServices::ReadNodesTensors(std::vector<std::string> name, std::vector<std::string> *ret_name,
                                     std::vector<char *> *data_ptr, std::vector<unsigned int> *data_size,
                                     std::vector<TypePtr> *dtype, std::vector<std::vector<int>> *shape) {
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

bool DebugServices::IsWatchPoint(std::string kernel_name,
                                 std::unordered_map<unsigned int, watchpoint_t> watchpoint_table) {
  bool ret = false;
  for (auto w_table_item : watchpoint_table) {
    auto check_node_list = std::get<1>(w_table_item).check_node_list;
    for (auto check_node : check_node_list) {
      std::string w_name = std::get<0>(check_node);
      bool w_type = std::get<1>(check_node);
      if ((w_type == true &&
           ((kernel_name.find(w_name) != string::npos && kernel_name.rfind(w_name, 0) == 0) || w_name == "*")) ||
          (w_type == false && kernel_name == w_name)) {
        ret = true;
        return ret;
      }
    }
  }
  return ret;
}

TensorLoader *DebugServices::tensor_loader() const { return tensor_loader_; }
std::unordered_map<unsigned int, DebugServices::watchpoint_t> DebugServices::GetWatchpointTable() {
  return watchpoint_table;
}

}  // namespace mindspore
