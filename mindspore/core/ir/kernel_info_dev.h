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

#ifndef MINDSPORE_CORE_IR_KERNEL_INFO_DEV_H_
#define MINDSPORE_CORE_IR_KERNEL_INFO_DEV_H_

#include <memory>
#include <map>
#include <utility>
#include <string>
#include "utils/info.h"
#include "utils/os.h"

namespace mindspore {
enum Axis : int {
  N = 0,
  C,
  H,
  W,
};

// Cache some runtime information which not be changed.
class RuntimeCache {
 public:
  std::pair<AnfNodePtr, size_t> get_prev_node_output(size_t index) {
    auto it = prev_node_output_map_.find(index);
    if (it != prev_node_output_map_.end()) {
      return it->second;
    } else {
      return std::pair<AnfNodePtr, size_t>();
    }
  }

  void set_prev_node_output(size_t index, std::pair<AnfNodePtr, size_t> output) {
    auto pr = std::make_pair(index, output);
    (void)prev_node_output_map_.insert(pr);
  }

  void update_prev_node_output(size_t index, const std::pair<AnfNodePtr, size_t> &output) {
    if (prev_node_output_map_.find(index) == prev_node_output_map_.end()) {
      MS_LOG(DEBUG) << "Index:" << index << " not in prev node map";
      return;
    }
    prev_node_output_map_[index] = output;
  }

  void reset() {
    MS_EXCEPTION_IF_CHECK_FAIL(!is_valid_, "this runtime cache is valid, can't reset!!!!");
    prev_node_output_map_.clear();
    device_target_.clear();
    output_tensor_num_ = -1;
    is_real_kernel_ = Uncached;
  }

  std::string device_target() { return device_target_; }

  void set_device_target(const std::string &target) { device_target_ = target; }
  bool is_valid() const { return is_valid_; }
  void set_is_valid(bool is_vaild) { is_valid_ = is_vaild; }
  void set_output_tensor_num(const ssize_t output_tensor_num) { output_tensor_num_ = output_tensor_num; }
  ssize_t output_tensor_num() const { return output_tensor_num_; }
  void set_real_kernel(CacheBool b) { is_real_kernel_ = b; }
  CacheBool is_real_kernel() const { return is_real_kernel_; }

 private:
  bool is_valid_{false};
  std::map<size_t, std::pair<AnfNodePtr, size_t>> prev_node_output_map_;
  std::string device_target_;
  ssize_t output_tensor_num_ = -1;
  CacheBool is_real_kernel_ = Uncached;
};
// Interface for device kernel program information.
class KernelInfoDevice {
 public:
  class RuntimeCacheScope {
   public:
    RuntimeCacheScope(RuntimeCache &base, std::mutex &mu) : runtime_cache_(base), mu_(mu) { mu_.lock(); }
    RuntimeCacheScope(const RuntimeCacheScope &other) = delete;
    RuntimeCacheScope operator=(const RuntimeCacheScope &other) = delete;
    ~RuntimeCacheScope() { mu_.unlock(); }
    RuntimeCache &runtime_cache() { return runtime_cache_; }

   private:
    RuntimeCache &runtime_cache_;
    std::mutex &mu_;
  };
  // If kernel program was built and build info is set.
  virtual bool has_build_info() const = 0;

  RuntimeCacheScope runtime_cache() { return RuntimeCacheScope(runtime_cache_, mu_); }

  virtual ~KernelInfoDevice() {}

 private:
  RuntimeCache runtime_cache_;
  std::mutex mu_;
};
using KernelInfoDevicePtr = std::shared_ptr<KernelInfoDevice>;
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_KERNEL_INFO_DEV_H_
