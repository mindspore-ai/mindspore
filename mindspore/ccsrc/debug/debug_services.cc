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

void DebugServices::add_watchpoint(unsigned int id, unsigned int watch_condition,
                                   const std::vector<std::tuple<std::string, bool>> &check_node_list) {
  std::lock_guard<std::mutex> lg(lock_);

  watchpoint_t watchpoint_item;

  watchpoint_item.id = id;

  if (watch_condition == 0) {
    watchpoint_item.conditions.nan.enabled = true;
  } else if (watch_condition == 1) {
    watchpoint_item.conditions.inf.enabled = true;
    watchpoint_item.conditions.neg_inf.enabled = true;
  }

  watchpoint_item.check_node_list = check_node_list;

  watchpoint_table[id] = watchpoint_item;
}

void DebugServices::remove_watchpoint(unsigned int id) {
  std::lock_guard<std::mutex> lg(lock_);
  watchpoint_table.erase(id);
}

void DebugServices::check_watchpoints(std::vector<std::string> *name, std::vector<std::string> *slot,
                                      std::vector<char *> *data_ptr, std::vector<unsigned int> *data_size,
                                      std::vector<int> *condition, std::vector<unsigned int> *wacthpoint_id) {
  std::lock_guard<std::mutex> lg(lock_);

  std::vector<std::shared_ptr<TensorData>> tensor_list = tensor_loader_->GetTensor();

  std::string current_tensor_name;
  std::unordered_map<unsigned int, watchpoint_t> watchpoints_to_check_table;

  for (std::size_t i = 0; i < tensor_list.size(); i++) {
    current_tensor_name = tensor_list[i]->GetName();
    mindspore::tensor::TensorPtr tensor_ptr = tensor_list[i]->GetTensor();
    int tensor_data_type = tensor_ptr->data_type_c();

    // check if we need to analyze this node and for which watchpoints we will check
    // create a list of watchpoints to check
    watchpoints_to_check_table.clear();
    for (auto w_table_item : watchpoint_table) {
      // if the watchpoint is checking for a nan or inf and the current tensor is not of a float type, then
      // don't check the watchpoint for this tensor
      if (std::get<1>(w_table_item).conditions.inf.enabled || std::get<1>(w_table_item).conditions.neg_inf.enabled ||
          std::get<1>(w_table_item).conditions.nan.enabled) {
        if (tensor_data_type != kNumberTypeFloat16 && tensor_data_type != kNumberTypeFloat &&
            tensor_data_type != kNumberTypeFloat32 && tensor_data_type != kNumberTypeFloat64) {
          continue;
        }
      }

      auto check_node_list = std::get<1>(w_table_item).check_node_list;

      for (auto check_node : check_node_list) {
        std::string w_name = std::get<0>(check_node);
        bool w_type = std::get<1>(check_node);

        // check if the current node tensor name is included the watchpoint
        std::string current_node_name = current_tensor_name.substr(0, current_tensor_name.find_first_of(":"));
        if ((w_type == true && (current_tensor_name.find(w_name) != string::npos || w_name == "*")) ||
            (w_type == false && current_node_name == w_name)) {
          watchpoints_to_check_table[w_table_item.second.id] = w_table_item.second;
          break;
        }
      }
    }

    // check if no watchpoints are valid for the current tensor
    if (watchpoints_to_check_table.empty()) {
      continue;
    }

    // need to add support for float16 and float64, and other types when we support conditions beyond inf and nan
    if (tensor_data_type != kNumberTypeFloat && tensor_data_type != kNumberTypeFloat32) {
      continue;
    }

    float *start_addr = reinterpret_cast<float *>(tensor_ptr->data_c());
    unsigned int num_elements = (tensor_ptr->data().nbytes()) / sizeof(float);

    std::unordered_map<unsigned int, watchpoint_t>::iterator it_w_table_check;
    std::vector<unsigned int> hit_encountered;

    for (unsigned int index = 0; index < num_elements; index++) {
      float x = start_addr[index];
      it_w_table_check = watchpoints_to_check_table.begin();

      while (it_w_table_check != watchpoints_to_check_table.end()) {
        if ((it_w_table_check->second.conditions.inf.enabled || it_w_table_check->second.conditions.neg_inf.enabled) &&
            isinf(x)) {
          hit_encountered.push_back(it_w_table_check->second.id);
        } else if (it_w_table_check->second.conditions.nan.enabled && isnan(x)) {
          hit_encountered.push_back(it_w_table_check->second.id);
        }

        ++it_w_table_check;
      }

      if (hit_encountered.size()) {
        for (auto it_hit_id = hit_encountered.begin(); it_hit_id != hit_encountered.end(); ++it_hit_id) {
          std::string name_no_slot = current_tensor_name.substr(0, current_tensor_name.find_first_of(":"));
          name->push_back(name_no_slot);

          slot->push_back(std::to_string(tensor_list[i]->GetSlot()));
          data_ptr->push_back(reinterpret_cast<char *>(tensor_ptr->data_c()));
          data_size->push_back(tensor_ptr->data().nbytes());

          int condition_item = -1;
          if (watchpoint_table[*it_hit_id].conditions.nan.enabled) {
            condition_item = 0;
          } else if (watchpoint_table[*it_hit_id].conditions.inf.enabled ||
                     watchpoint_table[*it_hit_id].conditions.neg_inf.enabled) {
            condition_item = 1;
          }
          condition->push_back(condition_item);

          wacthpoint_id->push_back(*it_hit_id);

          watchpoints_to_check_table.erase(*it_hit_id);
        }

        hit_encountered.clear();
      }

      if (watchpoints_to_check_table.empty()) {
        break;
      }
    }
  }
}

void DebugServices::read_nodes_tensors(std::vector<std::string> name, std::vector<std::string> *ret_name,
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

TensorLoader *DebugServices::get_tensor_loader() const { return tensor_loader_; }

}  // namespace mindspore
