/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "debug/tensor_data.h"
#ifdef ONLINE_DBG_MODE
#include "debug/data_dump/dump_json_parser.h"
#endif
namespace mindspore {
class TensorLoader {
 public:
  TensorLoader() : mem_total_(0), mem_usage_(0) {}

  ~TensorLoader() { EmptyTensor(); }

  void MoveTensorCurrentToPrev(const std::string &tensor_name) {
    auto handle = tensor_list_map_.extract(tensor_name);
    if (!handle.empty()) {
      MS_LOG(INFO) << "Moving " << tensor_name << " from current map to previous map";
      prev_tensor_list_map_.insert(std::move(handle));
    }
  }

  void SwapCurrentPrev() { tensor_list_map_.swap(prev_tensor_list_map_); }

  bool TensorExistsInCurrent(const std::string &tensor_name) const {
    return tensor_list_map_.find(tensor_name) != tensor_list_map_.end();
  }

  // only parameters will return true
  bool PrevTensorExistsInCurrent(const std::string &tensor_name) const {
    return TensorExistsInCurrent(tensor_name + ":prev");
  }

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
    if (tensor_name.length() <= suffix.length()) {
      return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), tensor_name.rbegin());
  }

  /*
   * Feature group: Dump, Online debugger and Offline debugger.
   * Target device group: Ascend, GPU.
   * Runtime category: Old runtime, MindRT.
   * Description: Load new tensor into tensor_list_map_ (debugger backend cache). In offline debugger, add ":prev" to
   * the previous tensor's name to avoid segfault caused by wrongly evicting the tensor when memory limit is enabled.
   */
  bool LoadNewTensor(const std::shared_ptr<TensorData> &tensor, bool keep_prev) {
    std::lock_guard<std::mutex> lg(lock_);
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
    std::string output_type = tensor->GetIsOutput() ? "1" : "0";
    key_name += (":" + std::to_string(tensor->GetDeviceId()) + ":" + std::to_string(tensor->GetRootGraphId()) + ":" +
                 output_type + ":" + std::to_string(tensor->GetSlot()));
    if (tensor_list_map_.find(key_name) != tensor_list_map_.end() &&
        tensor->GetIteration() == tensor_list_map_[key_name]->GetPrevIteration()) {
      key_name += ":prev";
    }
#endif
    tensor_list_map_[key_name] = tensor;  // use [] instead of insert to ensure latest value
    return true;
  }

  std::vector<std::shared_ptr<TensorData>> GetTensor() {
    std::vector<std::shared_ptr<TensorData>> tensor_list;
    for (auto it = tensor_list_map_.cbegin(); it != tensor_list_map_.cend(); ++it) {
      if (!IsPrevTensor(it->first)) {
        tensor_list.push_back(it->second);
      }
    }
    return tensor_list;
  }

  std::shared_ptr<TensorData> GetTensor(const std::string &tensor_name) const {
    auto iter = tensor_list_map_.find(tensor_name);
    if (iter != tensor_list_map_.end()) {
      return iter->second;
    }
    return nullptr;
  }

  std::shared_ptr<TensorData> GetPrevTensor(const std::string &tensor_name) {
    if (tensor_list_map_.find(tensor_name + ":prev") != tensor_list_map_.end()) {
      return tensor_list_map_[tensor_name + ":prev"];
    }
    return nullptr;
  }

  /*
   * Feature group: Online debugger.
   * Target device group: Ascend, GPU.
   * Runtime category: Old runtime, MindRT.
   * Description: Search and obtain TensorData for a list of tensors from tensor_list_map_ (debugger backend cache).
   * Return nullptr if the tensor is not found.
   */
  void SearchTensors(const std::vector<std::string> &search_list,
                     std::vector<std::tuple<std::string, std::shared_ptr<TensorData>>> *result_list) {
    for (auto i : search_list) {
      std::map<std::string, std::shared_ptr<TensorData>>::const_iterator iter = tensor_list_map_.find(i);
      if (iter != tensor_list_map_.cend()) {
        result_list->push_back(std::make_tuple(i, iter->second));
      } else {
        result_list->push_back(std::make_tuple(i, nullptr));
      }
    }
  }

  void EmptyTensor() noexcept {
    std::lock_guard<std::mutex> lg(lock_);
    prev_tensor_list_map_.clear();
    tensor_list_map_.swap(prev_tensor_list_map_);
  }

  void EmptyCurrentTensor() { tensor_list_map_.clear(); }

  bool EnableMemoryControl() const { return mem_total_ > 0; }

  /*
   * Feature group: Offline debugger.
   * Target device group: Ascend, GPU.
   * Runtime category: Old runtime, MindRT.
   * Description: This function is for memory control feature only. When finishing using a tensor in offline debugger,
   * it will be added to cache_evict_queue_ and become an eviction candidate. Once there is no memory to read in a new
   * tensor, it will be evicted from cache.
   */
  void AppendToCacheEvictQueue(const std::string &tensor_name) {
    std::lock_guard<std::mutex> lk(mem_lock_);
    if (std::find(cache_evict_queue_.begin(), cache_evict_queue_.end(), tensor_name) == cache_evict_queue_.end()) {
      cache_evict_queue_.push_back(tensor_name);
      evict_cond.notify_one();
    }
  }

  /*
   * Feature group: Offline debugger.
   * Target device group: Ascend, GPU.
   * Runtime category: Old runtime, MindRT.
   * Description: This function is for memory control feature only. Check if the tensor size is greater than the preset
   * limit. If not, evect the candidate tensor in cache_evict_queue_ to make room for it.
   */
  bool CheckMemoryAvailable(const std::string &backend_name, const uint64_t data_size) {
    // 1. Check if the tensor can fit in the entire limit. If not, don't attempt any read or evictions and generate
    // warning.
    if (data_size > mem_total_) {
      MS_LOG(ERROR) << "Failed to load data of tensor " << backend_name << " because the its data size (" << data_size
                    << ") exceeds the maximum memory limit (" << mem_total_ << ").";
      return false;
    }
    // 2. Check if there's is enough cache space available for current tensor. If not, try evict cache.
    bool ret = CheckAndEvictTensorCache(data_size);
    return ret;
  }

  /*
   * Feature group: Offline debugger.
   * Target device group: Ascend, GPU.
   * Runtime category: Old runtime, MindRT.
   * Description: This function is for memory control feature only. Greedily evict not-in-use tensors from cache queue.
   * If no candidate in the queue, block the thread until there is any candidate available.
   */
  bool CheckAndEvictTensorCache(const uint64_t data_size) {
    std::string candidate_name;
    uint64_t candidates_size;
    std::unique_lock<std::mutex> lk(mem_lock_);
    while (data_size > mem_total_ - mem_usage_) {
      // wait until there is any not-in-use candidate to be evicted from cache
      evict_cond.wait(lk, [this] { return !cache_evict_queue_.empty(); });
      candidate_name = cache_evict_queue_.front();
      cache_evict_queue_.pop_front();
      // evict candidate tensor
      auto tensor = GetTensor(candidate_name);
      if (tensor == nullptr) {
        MS_LOG(INFO) << "Tensor: " << candidate_name << " has already been evicted.";
        lock_.unlock();
        continue;
      }
      candidates_size = tensor->GetByteSize();
      tensor_list_map_.erase(candidate_name);
      mem_usage_ = std::max(uint64_t(0), mem_usage_ - candidates_size);
      MS_LOG(INFO) << "Evict tensor: " << candidate_name;
    }
    // Reserve space for the current target tensor.
    mem_usage_ = std::min(mem_total_, mem_usage_ + data_size);
    return true;
  }

  void SetMemTotal(uint64_t total_mem_size) { this->mem_total_ = total_mem_size; }

#ifdef ONLINE_DBG_MODE
  /*
   * Feature group: Dump.
   * Target device group: GPU, Ascend.
   * Runtime category: Old runtime, MindRT.
   * Description: Load tensor data from debugger backend cache (tensor_list_map_) and dump to file in npy format,
   *              used for GPU and Ascend KernelByKernel mode.
   */
  bool DumpTensorToFile(const std::string &filepath, const std::string &tensor_name, size_t slot) {
    if (filepath.empty()) {
      MS_LOG(ERROR) << "Dump file path is null!";
      return false;
    }

    std::string tensor_loader_name = tensor_name + ":" + std::to_string(slot);
    std::map<std::string, std::shared_ptr<TensorData>>::const_iterator iter = tensor_list_map_.find(tensor_loader_name);
    if (iter != tensor_list_map_.cend()) {
      std::shared_ptr<TensorData> node = iter->second;
      std::string path = filepath + '.' + node->GetFormat();
      if (node->GetByteSize() == 0) {
        MS_LOG(INFO) << "The byte size is 0 for tensor: " << tensor_loader_name;
        return false;
      }
      return DumpJsonParser::DumpToFile(path, node->GetDataPtr(), node->GetByteSize(), node->GetShape(),
                                        StringToTypeId(node->GetTypeString()));
    }
    MS_LOG(INFO) << "Tensor name:" << tensor_name << " not found in tensor_list_map_";
    return false;
  }
#endif

 private:
  // the pair is (device_id, iteration)
  std::map<std::string, std::shared_ptr<TensorData>> tensor_list_map_;
  std::map<std::string, std::shared_ptr<TensorData>> prev_tensor_list_map_;
  std::mutex lock_;
  std::mutex mem_lock_;
  uint64_t mem_total_;
  uint64_t mem_usage_;
  std::deque<std::string> cache_evict_queue_;
  std::condition_variable evict_cond;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_TENSOR_LOAD_H_
