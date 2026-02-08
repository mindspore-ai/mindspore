/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_AKG_AKG_UTILS_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_AKG_AKG_UTILS_H_
#include <string>
#include <set>
#include <memory>
#include <vector>
#include <utility>
#include <map>
#include <tuple>
#include <unordered_map>
#include <mutex>
#include "nlohmann/json.hpp"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/kash/kernel_pack.h"

namespace mindspore {
namespace kernel {
namespace akg {
using std::string;
using std::vector;

struct KernelMetaInfo {
  uint32_t block_dim_;
  void *launch_func_;
  void *handle_;
};
using KernelMetaPtr = std::shared_ptr<KernelMetaInfo>;

class KernelManager {
 public:
  KernelManager() = default;
  ~KernelManager();
  void *GenFuncStub(const KernelPack &kernel_pack, bool force_reload, uint32_t *block_dim, void **handle);

 private:
  void GetFunctionAndKernelName(const std::string &fn, const std::string &kernel_name, std::string *fn_so,
                                std::string *fn_kernel);
  static std::unordered_map<string, KernelMetaPtr> info_table_;
  static std::atomic<uintptr_t> kernel_stub_gen_;
  static std::mutex info_table_mutex_;
};
}  // namespace akg
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_AKG_AKG_UTILS_H_
