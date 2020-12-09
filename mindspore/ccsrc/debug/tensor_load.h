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
#include <mutex>
#include <tuple>
#include <string>
#include <utility>
#include "debug/tensor_data.h"
#include "debug/data_dump/dump_json_parser.h"
#include "ir/dtype.h"
namespace mindspore {
class TensorLoader {
 public:
  TensorLoader() : iter_num(-1) {}

  ~TensorLoader() { EmptyTensor(); }

  void MoveTensorCurrentToPrev(std::string tensor_name) {
    auto handle = tensor_list_map.extract(tensor_name);
    if (!handle.empty()) {
      MS_LOG(INFO) << "Moving " << tensor_name << " from current map to previous map";
      prev_tensor_list_map.insert(std::move(handle));
    }
  }

  void SwapCurrentPrev() { tensor_list_map.swap(prev_tensor_list_map); }

  bool TensorExistsInCurrent(std::string tensor_name) {
    return tensor_list_map.find(tensor_name) != tensor_list_map.end();
  }

  // only parameters will return true
  bool PrevTensorExistsInCurrent(std::string tensor_name) { return TensorExistsInCurrent(tensor_name + ":prev"); }

  void MoveParametersCurrentToPrev() {
    MS_LOG(INFO) << "Moving parameters from current map to previous map";
    auto iter = tensor_list_map.begin();
    while (iter != tensor_list_map.end()) {
      auto key = iter->first;
      if (PrevTensorExistsInCurrent(key)) {
        // :prev tensor only exists for parameter. Move it to prev
        ++iter;
        MoveTensorCurrentToPrev(key);
      } else {
        ++iter;
      }
    }
  }

  bool IsPrevTensor(std::string tensor_name) {
    const std::string suffix = ":prev";
    if (tensor_name.length() <= suffix.length()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), tensor_name.rbegin());
  }

  bool LoadNewTensor(std::shared_ptr<TensorData> tensor, bool keep_prev) {
    std::lock_guard<std::mutex> lg(lock_);
    if (keep_prev) {
      // add prev step tensor into current step map with ":prev" suffix
      auto handle = prev_tensor_list_map.extract(tensor->GetName());
      if (!handle.empty()) {
        handle.key() = tensor->GetName() + ":prev";
        tensor_list_map.insert(std::move(handle));
      }
    }
    tensor_list_map[tensor->GetName()] = tensor;  // use [] instead of insert to ensure latest value
    auto node_name = tensor->GetName();
    node_name = node_name.substr(0, node_name.find_first_of(":"));
    node_tensor_map.insert({node_name, tensor});
    return true;
  }

  std::vector<std::shared_ptr<TensorData>> GetTensor() {
    std::vector<std::shared_ptr<TensorData>> tensor_list;
    for (auto &it : tensor_list_map) {
      if (!IsPrevTensor(it.first)) tensor_list.push_back(it.second);
    }
    return tensor_list;
  }

  std::shared_ptr<TensorData> GetTensor(const std::string &tensor_name) {
    auto iter = tensor_list_map.find(tensor_name);
    if (iter != tensor_list_map.end()) return iter->second;
    return nullptr;
  }

  uint32_t GetIterNum() { return iter_num; }

  std::map<std::string, std::shared_ptr<TensorData>> GetTensorMap() { return tensor_list_map; }

  std::shared_ptr<TensorData> GetPrevTensor(const std::string &tensor_name) {
    if (tensor_list_map.find(tensor_name + ":prev") != tensor_list_map.end()) {
      return tensor_list_map[tensor_name + ":prev"];
    }
    return nullptr;
  }

  std::vector<std::shared_ptr<TensorData>> GetNodeTensorMap(std::string node_name) {
    std::vector<std::shared_ptr<TensorData>> tensors;
    for (auto itr = node_tensor_map.begin(); itr != node_tensor_map.end(); itr++) {
      if (itr->first == node_name) {
        tensors.push_back(itr->second);
      }
    }
    return tensors;
  }

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

  void EmptyTensor() {
    std::lock_guard<std::mutex> lg(lock_);
    prev_tensor_list_map.clear();
    node_tensor_map.clear();
    tensor_list_map.swap(prev_tensor_list_map);
  }

  void EmptyPrevTensor() { prev_tensor_list_map.clear(); }

  void EmptyCurrentTensor() {
    tensor_list_map.clear();
    node_tensor_map.clear();
  }

  void set_iter_num(uint32_t iter_num) { this->iter_num = iter_num; }

  bool DumpTensorToFile(const std::string &tensor_name, bool trans_flag, const std::string &filepath,
                        const std::string &host_fmt, const std::vector<int64_t> &host_shape, TypeId host_type,
                        TypeId addr_type_id, const std::string &addr_format, size_t slot) const {
    if (filepath.empty()) {
      MS_LOG(ERROR) << "Dump file path is null!";
      return false;
    }
    std::string shape = "shape";
    if (host_shape.size()) {
      for (auto &value : host_shape) {
        shape = shape + '_' + std::to_string(value);
      }
    } else {
      shape = shape + "_0";
    }
    std::string file_extension = ".bin";
    std::string path = "";
    if (trans_flag) {
      path = filepath + '_' + shape + '_' + TypeIdLabel(host_type) + '_' + host_fmt + file_extension;
    } else {
      path = filepath + '_' + shape + '_' + TypeIdToType(addr_type_id)->ToString() + '_' + addr_format + file_extension;
    }

    MS_LOG(INFO) << "Dump path is " << path;

    std::string tensor_loader_name = tensor_name + ":" + std::to_string(slot);
    auto iter = tensor_list_map.find(tensor_loader_name);
    if (iter != tensor_list_map.end()) {
      std::shared_ptr<TensorData> node = iter->second;
      mindspore::tensor::TensorPtr out_tensor = node->GetTensor();
      size_t host_size = out_tensor->data().nbytes();

      return DumpJsonParser::DumpToFile(path, out_tensor->data_c(), host_size);
    }
    MS_LOG(INFO) << "Tensor name:" << tensor_name << " not found in tensor_list_map";
    return true;
  }

 private:
  std::map<std::string, std::shared_ptr<TensorData>> tensor_list_map;
  std::multimap<std::string, std::shared_ptr<TensorData>> node_tensor_map;
  std::map<std::string, std::shared_ptr<TensorData>> prev_tensor_list_map;
  uint32_t iter_num;
  std::mutex lock_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_TENSOR_LOAD_H_
