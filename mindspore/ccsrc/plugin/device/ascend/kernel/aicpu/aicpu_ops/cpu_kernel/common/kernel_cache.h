/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#ifndef AICPU_CONTEXT_COMMON_KERNEL_CACHE_H
#define AICPU_CONTEXT_COMMON_KERNEL_CACHE_H

#include <cstdint>

#include <list>
#include <memory>
#include <unordered_map>
#include <utility>
#include <mutex>

#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/common/kernel_log.h"
#include "cpu_kernel/common/device_cpu_kernel.h"

namespace aicpu {
template <class T>
class KernelCache {
 public:
  KernelCache() : sess_flag_(false), capacity_(1) {}
  virtual ~KernelCache() = default;

  /*
   * Init kernel cache.
   * @param sess_flag: whether it's a session scene, false need to support LRU
   * algorithm
   * @return int32_t: 0 indicates success, while the others fail
   */
  int32_t Init(bool sess_flag) {
    sess_flag_ = sess_flag;
    return InitParameter();
  }

  /*
   * run kernel.
   * @param param: kernel context
   * @return int32_t: 0 indicates success, whilWe the others fail
   */
  virtual int32_t RunKernel(void *param) = 0;

  /*
   * run kernel with blockDimInfo.
   * @param param: kernel context and kernel context and blkDimInfo
   * @return int32_t: 0 indicates success, whilWe the others fail
   */
  virtual int32_t RunCpuKernelWithBlock(void *param, struct BlkDimInfo *blkDimInfo) = 0;
  /*
   * get kernel cache, the lru algorithm is supported in non-session scenarios
   * @param key: kernel id
   * @return T *: cache content pointer
   */
  T *GetCache(uint64_t key) {
    KERNEL_LOG_DEBUG("GetCache begin, key[%llu].", key);
    T *ret = nullptr;
    std::unique_lock<std::mutex> lock(kernel_mutex_);
    auto it = kernel_cache_iter_.find(key);
    if (it != kernel_cache_iter_.end()) {
      KERNEL_LOG_DEBUG("GetCache success, key[%llu].", key);
      ret = it->second->second.get();
      if (!sess_flag_) {
        auto pair_iter = it->second;
        std::pair<uint64_t, std::shared_ptr<T>> pair = *pair_iter;
        kernel_cache_.erase(pair_iter);
        kernel_cache_.push_front(pair);
        kernel_cache_iter_[key] = kernel_cache_.begin();
      }
    }
    return ret;
  }

  /*
   * set kernel cache, the lru algorithm is supported in non-session scenarios
   * @param key: kernel id
   * @param value: cache content
   */
  void SetCache(uint64_t key, std::shared_ptr<T> value) {
    KERNEL_LOG_DEBUG("SetCache begin, key[%llu].", key);
    std::unique_lock<std::mutex> lock(kernel_mutex_);
    auto iter = kernel_cache_iter_.find(key);
    if (iter != kernel_cache_iter_.end()) {
      KERNEL_LOG_DEBUG("SetCache update cache, key[%llu].", key);
      auto pair_iter = iter->second;
      pair_iter->second = value;
      if (!sess_flag_) {
        std::pair<uint64_t, std::shared_ptr<T>> pair = *pair_iter;
        kernel_cache_.erase(pair_iter);
        kernel_cache_.push_front(pair);
        kernel_cache_iter_[key] = kernel_cache_.begin();
      }
    } else {
      std::pair<uint64_t, std::shared_ptr<T>> pair = std::make_pair(key, value);
      if ((capacity_ < kernel_cache_.size()) && (!sess_flag_)) {
        uint64_t del_key = kernel_cache_.back().first;
        KERNEL_LOG_DEBUG(
          "SetCache is full, pop last element, capacity[%u], delete "
          "key[%llu].",
          capacity_, key);
        kernel_cache_.pop_back();
        auto del_iter = kernel_cache_iter_.find(del_key);
        if (del_iter != kernel_cache_iter_.end()) {
          kernel_cache_iter_.erase(del_iter);
        }
      }
      KERNEL_LOG_DEBUG("SetCache success, key[%llu].", key);
      kernel_cache_.push_front(pair);
      kernel_cache_iter_[key] = kernel_cache_.begin();
    }
  }

  /*
   * get session flag, true means session scene
   * @return bool: whether it's a session scene
   */
  bool GetSessionFlag() const { return sess_flag_; }

  /*
   * get kernel cache capacity
   * @return uint32_t: lru capacity
   */
  uint32_t GetCapacity() { return capacity_; }

  /*
   * set kernel cache capacity
   * @param capacity: lru capacity
   */
  void SetCapacity(uint32_t capacity) { capacity_ = capacity; }

  /*
   * get all kernel cache
   * @return std::list<std::pair<uint64_t, std::shared_ptr<T>>>: all cache,
   * pair<kernel id, cache>
   */
  std::list<std::pair<uint64_t, std::shared_ptr<T>>> GetAllKernelCache() { return kernel_cache_; }

 protected:
  virtual int32_t InitParameter() = 0;

 private:
  KernelCache(const KernelCache &) = delete;
  KernelCache(KernelCache &&) = delete;
  KernelCache &operator=(const KernelCache &) = delete;
  KernelCache &operator=(KernelCache &&) = delete;

  bool sess_flag_;     // whether it's a session scene, false need to support LRU
  uint32_t capacity_;  // lru capacity
  std::mutex kernel_mutex_;
  std::list<std::pair<uint64_t, std::shared_ptr<T>>> kernel_cache_;  // all kernel cache, key is kernel id
  std::unordered_map<uint64_t, typename std::list<std::pair<uint64_t, std::shared_ptr<T>>>::iterator>
    kernel_cache_iter_;
};
}  // namespace aicpu
#endif  // AICPU_CONTEXT_COMMON_KERNEL_CACHE_H
