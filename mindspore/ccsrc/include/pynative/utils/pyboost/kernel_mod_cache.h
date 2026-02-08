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
#ifndef MINDSPORE_CCSRC_PYBOOST_KERNEL_MOD_CACHE_H_
#define MINDSPORE_CCSRC_PYBOOST_KERNEL_MOD_CACHE_H_

#include <utility>
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <unordered_map>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"

namespace mindspore {
namespace kernel {
class PYBOOST_API KernelModCache {
 public:
  static KernelModCache &GetInstance();

  std::string GetKernelModKey(const std::string &op_name, const std::string &device_name,
                              const std::vector<KernelTensor *> &inputs);
  std::string GetPrimAttrKernelModKey(const PrimitivePtr &prim, const std::string &device_name,
                                      const std::vector<KernelTensor *> &inputs);
  KernelModPtr GetKernelMod(const std::string &key);
  void SetCache(const std::string &key, const KernelModPtr &kernel_mod) { kernel_mod_cache_[key] = kernel_mod; }
  void ClearAllCache();
  void ClearOpCache(const std::string &key);

 private:
  KernelModCache() {}
  ~KernelModCache() = default;
  DISABLE_COPY_AND_ASSIGN(KernelModCache);
  mindspore::HashMap<std::string, KernelModPtr> kernel_mod_cache_;
};

}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PYBOOST_KERNEL_MOD_CACHE_H_
