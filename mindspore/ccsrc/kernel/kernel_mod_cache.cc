/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "kernel/kernel_mod_cache.h"

#include <vector>
#include <unordered_set>

namespace mindspore {
namespace kernel {
KernelModCache &KernelModCache::GetInstance() {
  static KernelModCache instance;
  return instance;
}

KernelModPtr KernelModCache::GetKernelMod(const std::string &key) {
  auto iter = kernel_mod_cache_.find(key);
  if (iter != kernel_mod_cache_.end()) {
    return iter->second;
  }
  return nullptr;
}

std::string KernelModCache::GetKernelModKey(const std::string &op_name, const std::string &device_name,
                                            const std::vector<KernelTensor *> &inputs) {
  std::string key = op_name;
  key += "_";
  key += device_name;
  for (const auto &input : inputs) {
    key += "_";
    key += std::to_string(input->dtype_id());
  }
  return key;
}
void KernelModCache::ClearAllCache() { kernel_mod_cache_.clear(); }
void KernelModCache::ClearOpCache(const std::string &key) { (void)kernel_mod_cache_.erase(key); }
}  // namespace kernel
}  // namespace mindspore
