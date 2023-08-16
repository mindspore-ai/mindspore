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
#ifndef MINDSPORE_CCSRC_KERNEL_FRAMEWORK_UTILS_H_
#define MINDSPORE_CCSRC_KERNEL_FRAMEWORK_UTILS_H_

#include <set>
#include <map>
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include <unordered_map>
#include "include/common/utils/utils.h"
#include "kernel/kernel.h"
#include "kernel/oplib/opinfo.h"
#include "kernel/kash/kernel_pack.h"
#include "kernel/kernel_build_info.h"
#include "include/backend/device_address.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace kernel {
constexpr auto kAkgKernelMeta = "akg_kernel_meta/";
constexpr auto kKernelMetaSuffix = "_kernel_meta/";
constexpr auto kJsonSuffix = ".json";
constexpr auto kInfoSuffix = ".info";

class BACKEND_EXPORT KernelMeta {
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

std::set<int64_t> GetShapeSetFromResizeMap(const CNodePtr &node);

BACKEND_EXPORT std::string GetCompilerCachePath();
bool CheckCache(const std::string &kernel_name);
KernelPackPtr SearchCache(const std::string &kernel_name, const std::string &processor);
KernelPackPtr InsertCache(const std::string &kernel_name, const std::string &processor);

BACKEND_EXPORT bool GetShapeSize(const ShapeVector &shape, const TypePtr &type_ptr, int64_t *size_i);

BACKEND_EXPORT bool ParseMetadata(const CNodePtr &kernel_node, const std::shared_ptr<const OpInfo> &op_info_ptr,
                                  Processor processor,
                                  std::vector<std::shared_ptr<KernelBuildInfo>> *const kernel_info_list);

BACKEND_EXPORT void SaveJsonInfo(const std::string &json_name, const std::string &info, const std::string &base_path);

std::string GetProcessor(const AnfNodePtr &anf_node);
Processor GetProcessor(const string &processor);
Processor GetProcessorFromContext();
std::string GetStrProcessorFromContext();

BACKEND_EXPORT std::vector<std::pair<AnfNodePtr, size_t>> GetOutputIndex(const std::vector<AnfNodePtr> &node_list,
                                                                         const std::vector<AnfNodePtr> &input_list,
                                                                         const std::vector<AnfNodePtr> &output_list);
BACKEND_EXPORT void GetValidKernelNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *node_list);
BACKEND_EXPORT void GetValidKernelNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *node_list,
                                        std::vector<AnfNodePtr> *input_list, std::vector<AnfNodePtr> *output_list);
void GetFuncGraphOutputNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *output_list);
void GetGraphRealOutput(const FuncGraphPtr &func_graph, std::vector<std::pair<AnfNodePtr, size_t>> *node_list);
BACKEND_EXPORT bool IsWeightBoundary(const AnfNodePtr &node);
BACKEND_EXPORT std::vector<int64_t> GetReduceAttrAxis(const CNodePtr &cnode);

struct KernelArgs {
  std::vector<KernelTensorPtr> inputs;
  std::vector<KernelTensorPtr> outputs;
  std::map<uint32_t, tensor::TensorPtr> depend_tensor_map;  // dynamic shape kernel may need this map
  // cppcheck-suppress unusedStructMember
  constexpr static char key[] = "KernelArgs";
};
BACKEND_EXPORT KernelArgs AbstractArgsFromCNode(const CNodePtr &cnode);
BACKEND_EXPORT KernelArgs AbstractArgsFromDeviceAddress(
  KernelMod *const kernel_mod, const std::vector<device::DeviceAddressPtr> &inputs_device_address,
  const std::vector<device::DeviceAddressPtr> &outputs_device_address, const AbstractBasePtr &abstract);
BACKEND_EXPORT std::shared_ptr<KernelArgs> GetArgsFromCNode(const CNodePtr &cnode);
BACKEND_EXPORT void SetArgsToCNode(const CNodePtr &cnode, const KernelArgs &args);

BACKEND_EXPORT BaseOperatorPtr CreateOperatorByCNode(const CNodePtr &cnode);
BACKEND_EXPORT void UpdateNodeShape(const CNodePtr &cnode);

BACKEND_EXPORT void SetInputsByDependMap(const std::map<uint32_t, tensor::TensorPtr> &depend_tensor_map,
                                         std::vector<KernelTensorPtr> *inputs, bool is_stored_in_device = false);
BACKEND_EXPORT void SetInputsByConstInputs(const CNodePtr &node,
                                           std::map<uint32_t, tensor::TensorPtr> *inputs_tensor_map);
BACKEND_EXPORT bool IfNeedSkipResize(const CNodePtr &node);

inline std::map<uint32_t, tensor::TensorPtr> GetKernelDepends(const CNodePtr &cnode) {
  auto args = GetArgsFromCNode(cnode);
  if (args) {
    return args->depend_tensor_map;
  }
  return std::map<uint32_t, tensor::TensorPtr>();
}

KernelObjectType StringToKernelObjectType(const std::string &object_type);

BACKEND_EXPORT void UnfoldKernelBuildInfo(const CNodePtr &kernel_node);
BACKEND_EXPORT int64_t CalOutputTupleSize(const AnfNodePtr &node);
BACKEND_EXPORT void SetDynamicInputSizeAttr(const CNodePtr &cnode);
BACKEND_EXPORT bool IsDynamicParamKernel(const std::string &op_name);
BACKEND_EXPORT std::pair<std::string, ExceptionType> KernelObjectTypeNotSupportWarning(const CNodePtr &kernel_node);
BACKEND_EXPORT bool IsKernelObjectTypeNotSupportedError(const std::string &error_str);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_FRAMEWORK_UTILS_H_
