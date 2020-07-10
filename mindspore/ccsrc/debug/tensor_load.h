/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_DEBUG_TENSOR_LOAD_H_
#define MINDSPORE_CCSRC_DEBUG_TENSOR_LOAD_H_

#include <memory>
#include <vector>
#include <map>
#include <tuple>
#include <string>
#include <utility>
#include "debug/tensor_data.h"
namespace mindspore {
class TensorLoader {
 public:
  TensorLoader() : iter_num(-1) {}

  ~TensorLoader() {}

  bool LoadNewTensor(std::shared_ptr<TensorData> tensor, bool keep_prev) {
    if (keep_prev) {
      // add prev step tensor into current step map with ":prev" suffix
      auto handle = prev_tensor_list_map.extract(tensor->GetName());
      if (!handle.empty()) {
        handle.key() = tensor->GetName() + ":prev";
        tensor_list_map.insert(std::move(handle));
      }
    }
    tensor_list.push_back(tensor);
    tensor_list_map.insert({tensor->GetName(), tensor});
    return true;
  }
  std::vector<std::shared_ptr<TensorData>> GetTensor() { return tensor_list; }

  uint32_t GetIterNum() { return iter_num; }

  std::map<std::string, std::shared_ptr<TensorData>> GetTensorMap() { return tensor_list_map; }
  void SearchTensors(const std::vector<std::string> &search_list,
                     std::vector<std::tuple<std::string, std::shared_ptr<TensorData>>> *result_list) {
    for (auto i : search_list) {
      std::map<std::string, std::shared_ptr<TensorData>>::iterator iter;
      iter = tensor_list_map.find(i);
      if (iter != tensor_list_map.end()) {
        result_list->push_back(std::make_tuple(i, iter->second));
      } else {
        result_list->push_back(std::make_tuple(i, nullptr));
      }
    }
  }

  bool EmptyTensor() {
    prev_tensor_list_map.clear();
    tensor_list_map.swap(prev_tensor_list_map);
    tensor_list.clear();
    return true;
  }

  void EmptyPrevTensor() { prev_tensor_list_map.clear(); }

  void set_iter_num(uint32_t iter_num) { this->iter_num = iter_num; }

 private:
  std::vector<std::shared_ptr<TensorData>> tensor_list;
  std::map<std::string, std::shared_ptr<TensorData>> tensor_list_map;
  std::map<std::string, std::shared_ptr<TensorData>> prev_tensor_list_map;
  uint32_t iter_num;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_TENSOR_LOAD_H_
