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
#include <deque>
#include <algorithm>
#ifdef OFFLINE_DBG_MODE
#include "debugger/offline_debug/offline_logger.h"
#endif
#include "debug/tensor_data.h"
#ifdef ONLINE_DBG_MODE
#include "debug/data_dump/dump_json_parser.h"
namespace mindspore {
#endif
class TensorLoader {
 public:
  TensorLoader() : iter_num_(-1), mem_total_(0), mem_usage_(0) {}

  ~TensorLoader() { EmptyTensor(); }

  void MoveTensorCurrentToPrev(std::string tensor_name) {
    auto handle = tensor_list_map_.extract(tensor_name);
    if (!handle.empty()) {
      MS_LOG(INFO) << "Moving " << tensor_name << " from current map to previous map";
      prev_tensor_list_map_.insert(std::move(handle));
    }
  }

  void SwapCurrentPrev() { tensor_list_map_.swap(prev_tensor_list_map_); }

  bool TensorExistsInCurrent(std::string tensor_name) const {
    return tensor_list_map_.find(tensor_name) != tensor_list_map_.end();
  }

  // only parameters will return true
  bool PrevTensorExistsInCurrent(std::string tensor_name) const { return TensorExistsInCurrent(tensor_name + ":prev"); }

  void MoveParametersCurrentToPrev() {
    MS_LOG(INFO) << "Moving parameters from current map to previous map";
    auto iter = tensor_list_map_.begin();
    while (iter != tensor_list_map_.end()) {
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

  bool IsPrevTensor(std::string tensor_name) const {
    const std::string suffix = ":prev";
    if (tensor_name.length() <= suffix.length()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), tensor_name.rbegin());
  }

  bool LoadNewTensor(std::shared_ptr<TensorData> tensor, bool keep_prev) {
    lock_.lock();
    auto tensor_name = tensor->GetName();
    if (keep_prev) {
      // add prev step tensor into current step map with ":prev" suffix
      auto handle = prev_tensor_list_map_.extract(tensor_name);
      if (!handle.empty()) {
        handle.key() = tensor_name + ":prev";
        tensor_list_map_.insert(std::move(handle));
      }
    }
    std::string key_name = tensor_name;
#ifdef OFFLINE_DBG_MODE
    key_name += (":" + std::to_string(tensor->GetDeviceId()) + ":" + std::to_string(tensor->GetRootGraphId()) + ":" +
                 std::to_string(tensor->GetIsOutput()) + ":" + std::to_string(tensor->GetSlot()));
    if (tensor_list_map_.find(key_name) != tensor_list_map_.end() &&
        tensor->GetIteration() == tensor_list_map_[key_name]->GetIteration() - 1) {
      key_name += ":prev";
    }
    lock_.unlock();
    AppendToCacheRecord(key_name, tensor->GetByteSize());
    lock_.lock();
    auto iter = tensor_list_map_.find(key_name);
    if (iter != tensor_list_map_.end()) {
      iter->second->DeleteDataPtr();
    }
#endif
    tensor_list_map_[key_name] = tensor;  // use [] instead of insert to ensure latest value
    lock_.unlock();
    return true;
  }

  std::vector<std::shared_ptr<TensorData>> GetTensor() {
    std::vector<std::shared_ptr<TensorData>> tensor_list;
    for (auto &it : tensor_list_map_) {
      if (!IsPrevTensor(it.first)) tensor_list.push_back(it.second);
    }
    return tensor_list;
  }

  std::shared_ptr<TensorData> GetTensor(const std::string &tensor_name) const {
    auto iter = tensor_list_map_.find(tensor_name);
    if (iter != tensor_list_map_.end()) return iter->second;
    return nullptr;
  }

  uint32_t GetIterNum() const { return iter_num_; }

  std::map<std::string, std::shared_ptr<TensorData>> GetTensorMap() { return tensor_list_map_; }

  std::shared_ptr<TensorData> GetPrevTensor(const std::string &tensor_name) {
    if (tensor_list_map_.find(tensor_name + ":prev") != tensor_list_map_.end()) {
      return tensor_list_map_[tensor_name + ":prev"];
    }
    return nullptr;
  }

  void SearchTensors(const std::vector<std::string> &search_list,
                     std::vector<std::tuple<std::string, std::shared_ptr<TensorData>>> *result_list) {
    for (auto i : search_list) {
      std::map<std::string, std::shared_ptr<TensorData>>::iterator iter;
      iter = tensor_list_map_.find(i);
      if (iter != tensor_list_map_.end()) {
        result_list->push_back(std::make_tuple(i, iter->second));
      } else {
        result_list->push_back(std::make_tuple(i, nullptr));
      }
    }
  }

  void EmptyTensor() {
    std::lock_guard<std::mutex> lg(lock_);
    prev_tensor_list_map_.clear();
    tensor_list_map_.swap(prev_tensor_list_map_);
  }

  void EmptyCurrentTensor() { tensor_list_map_.clear(); }

  void set_iter_num(uint32_t iter_num) { this->iter_num_ = iter_num; }

  bool EnableMemoryControl() { return mem_total_ > 0; }

  void ReleaseInUsedStatus(const std::string &tensor_name) {
    if (cache_status_map_.find(tensor_name) != cache_status_map_.end()) {
      cache_status_map_[tensor_name] = CacheStatus::KNotInUsed;
    }
  }

  void AppendToCacheRecord(const std::string &tensor_name, const uint64_t data_size) {
    std::lock_guard<std::mutex> lk(mem_lock_);
    // Set the current tensor to in-use because it's ready to be read in.
    cache_status_map_[tensor_name] = CacheStatus::KInUsed;
    // Push the current target tensor into cache record.
    auto find_key = std::find(tensor_queue_.begin(), tensor_queue_.end(), tensor_name);
    if (find_key == tensor_queue_.end()) {
      tensor_queue_.push_back(tensor_name);
    }
  }

  bool CheckMemoryAvailable(const std::string &backend_name, const uint64_t data_size) {
    // 1. Check if the tensor can fit in the entire limit. If not, don't attempt any read or evictions and generate
    // warning.
    if (data_size > mem_total_) {
      MS_LOG(ERROR) << "Failed to load data of tensor " << backend_name << " because the its data size (" << data_size
                    << ") exceeds the maximum memory limit (" << mem_total_ << ").";
      return false;
    }
    // 2. Check if there's is enough cache space available for current tensor. If not, try evict cache.
    mem_lock_.lock();
    bool ret = CheckAndEvictTensorCache(data_size);
    if (ret) {
      // Reserve space for the current target tensor.
      mem_usage_ = std::min(mem_total_, mem_usage_ + data_size);
    }
    mem_lock_.unlock();
    return ret;
  }

  bool CheckAndEvictTensorCache(const uint64_t data_size) {
    if (mem_total_ >= mem_usage_ && data_size <= mem_total_ - mem_usage_) {
      return true;
    }
    // Calculate the total size of all the not-in-used tensor in the cache.
    uint64_t candidates_size = 0;
    std::vector<std::pair<std::string, uint32_t>> candidate_tensors;
    for (uint32_t i = 0; i < tensor_queue_.size(); i++) {
      if (candidates_size >= data_size) {
        break;
      }
      std::string tensor_name = tensor_queue_[i];
      auto tensor_status = cache_status_map_[tensor_name];
      if (tensor_status == CacheStatus::KInUsed) {
        continue;
      }
      lock_.lock();
      candidates_size += tensor_list_map_[tensor_name]->GetByteSize();
      lock_.unlock();
      candidate_tensors.push_back(std::make_pair(tensor_name, i));
    }
    // Evict the candidate tensors if they can make room for the current target tensor.
    if (candidates_size < data_size) {
      return false;
    } else {
      uint32_t cnt = 0;
      for (auto candidate_tensor : candidate_tensors) {
        auto key_to_evict = candidate_tensor.first;
        auto idx_to_evict = candidate_tensor.second;
        lock_.lock();
        tensor_list_map_[key_to_evict]->DeleteDataPtr();
        tensor_list_map_.erase(key_to_evict);
        lock_.unlock();
        tensor_queue_.erase(tensor_queue_.begin() + idx_to_evict - cnt);
        MS_LOG(INFO) << "Evict tensor: " << key_to_evict;
        cnt++;
      }
      mem_usage_ = std::max(uint64_t(0), mem_usage_ - candidates_size);
    }
    return true;
  }

  void SetMemTotal(uint64_t total_mem_size) { this->mem_total_ = total_mem_size; }

#ifdef ONLINE_DBG_MODE
  bool DumpTensorToFile(const std::string &tensor_name, bool trans_flag, const std::string &filepath,
                        const std::string &host_fmt, const std::vector<int64_t> &host_shape, TypeId host_type,
                        TypeId device_type, const std::string &addr_format, size_t slot) {
    if (filepath.empty()) {
      MS_LOG(ERROR) << "Dump file path is null!";
      return false;
    }
    std::string path = "";
    if (trans_flag) {
      path = filepath + '.' + host_fmt;
    } else {
      path = filepath + '.' + addr_format;
    }

    MS_LOG(INFO) << "Dump path is " << path;

    std::string tensor_loader_name = tensor_name + ":" + std::to_string(slot);
    auto iter = tensor_list_map_.find(tensor_loader_name);
    if (iter != tensor_list_map_.end()) {
      std::shared_ptr<TensorData> node = iter->second;
      size_t host_size = node->GetByteSize();

      return DumpJsonParser::DumpToFile(path, node->GetDataPtr(), host_size, host_shape, host_type);
    }
    MS_LOG(INFO) << "Tensor name:" << tensor_name << " not found in tensor_list_map_";
    return true;
  }
#endif

 private:
  // the pair is (device_id, iteration)
  std::map<std::string, std::shared_ptr<TensorData>> tensor_list_map_;
  std::map<std::string, std::shared_ptr<TensorData>> prev_tensor_list_map_;
  uint32_t iter_num_;
  std::mutex lock_;
  std::mutex mem_lock_;
  uint64_t mem_total_;
  uint64_t mem_usage_;
  std::deque<std::string> tensor_queue_;
  std::map<std::string, bool> cache_status_map_;

  enum CacheStatus { KNotInUsed = 0, KInUsed = 1 };
};
#ifdef ONLINE_DBG_MODE
}  // namespace mindspore
#endif
#endif  // MINDSPORE_CCSRC_DEBUG_TENSOR_LOAD_H_
