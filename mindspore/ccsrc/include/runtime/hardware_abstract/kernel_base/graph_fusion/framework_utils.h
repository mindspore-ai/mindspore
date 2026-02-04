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
#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_FRAMEWORK_UTILS_H_
#define MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_FRAMEWORK_UTILS_H_

#include <set>
#include <map>
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include <unordered_map>
#include "ir/dtype/tensor_type.h"
#include "include/utils/utils.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/oplib/opinfo.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/kash/kernel_pack.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_build_info.h"
#include "device_address/device_address.h"
#include "ops/base_operator.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "runtime/hardware_abstract/visible.h"

namespace mindspore {
namespace kernel {
constexpr auto kAkgKernelMeta = "akg_kernel_meta/";
constexpr auto kKernelMetaSuffix = "_kernel_meta/";
constexpr auto kJsonSuffix = ".json";
constexpr auto kInfoSuffix = ".info";

class RUNTIME_HARDWARE_EXPORT KernelMeta {
 public:
  KernelMeta() = default;
  void Initialize(const std::string &backend = "akg");
  std::string Search(const std::string &kernel_name) const;
  bool Insert(const std::string &kernel_name, const std::string &kernel_json);
  std::string kernel_meta_path() const { return kernel_meta_path_; }
  bool initialized() const { return initialized_; }
  static KernelMeta *GetInstance() {
    static KernelMeta kernel_meta;
    return &kernel_meta;
  }
  ~KernelMeta() = default;

 private:
  bool initialized_ = false;
  std::string kernel_meta_path_;
  std::unordered_map<std::string, std::string> kernel_meta_map_;
};

RUNTIME_HARDWARE_EXPORT std::string GetCompilerCachePath();

RUNTIME_HARDWARE_EXPORT void SaveJsonInfo(const std::string &json_name, const std::string &info,
                                          const std::string &base_path);

RUNTIME_HARDWARE_EXPORT std::string GetStrProcessorFromContext();

RUNTIME_HARDWARE_EXPORT void GetValidKernelNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *node_list);
RUNTIME_HARDWARE_EXPORT void GetValidKernelNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *node_list,
                                                 std::vector<AnfNodePtr> *input_list,
                                                 std::vector<AnfNodePtr> *output_list);
RUNTIME_HARDWARE_EXPORT void GetFuncGraphOutputNodes(const FuncGraphPtr &func_graph,
                                                     std::vector<AnfNodePtr> *output_list);

struct KernelArgs {
  std::vector<KernelTensorPtr> inputs;
  std::vector<KernelTensorPtr> outputs;
  std::map<uint32_t, tensor::TensorPtr> depend_tensor_map;  // dynamic shape kernel may need this map
  // cppcheck-suppress unusedStructMember
  constexpr static char key[] = "KernelArgs";
};

RUNTIME_HARDWARE_EXPORT bool CheckResizeCondition(const CNodePtr &node);
RUNTIME_HARDWARE_EXPORT bool IsDynamicParamKernel(const std::string &op_name);
RUNTIME_HARDWARE_EXPORT std::pair<std::vector<DataType>, std::vector<DataType>> GetInOutDataTypesFromKernelAttr(
  const KernelAttr &kernel_attr);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_FRAMEWORK_UTILS_H_
