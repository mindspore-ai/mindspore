/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_mod.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_compile.h"

namespace mindspore {
namespace kernel {
namespace ascend {
std::string TbeKernelCompileManager::TbeOpSelectFormat(const CNodePtr &node) const { return std::string(); }
bool TbeKernelCompileManager::TbeOpCheckSupported(const CNodePtr &node, nlohmann::json *kernel_json) const {
  return true;
}
TbeKernelCompileManager::~TbeKernelCompileManager() {}
bool TbeKernelCompileManager::tbe_init_flag_ = true;

void TbeKernelCompileManager::TbeInitialize() {}
// pre build
void TbeKernelCompileManager::TbePreBuild(const std::shared_ptr<session::KernelGraph> &kernel_graph) {}
// single op compile
std::pair<std::vector<CNodePtr>, std::vector<CNodePtr>> TbeKernelCompileManager::TbeSingleOpCompile(
  const std::vector<CNodePtr> &anf_nodes) {
  std::vector<CNodePtr> success_node;
  std::vector<CNodePtr> failed_node;
  return std::make_pair(success_node, failed_node);
}
// fusion op compile
JsonNameMap TbeKernelCompileManager::TbeFusionOpCompile(const std::vector<FusionScopeInfo> &fusion_scopes) {
  JsonNameMap json_name_map;
  for (const auto &fusion_scope_iter : fusion_scopes) {
    json_name_map[fusion_scope_iter.scope_id] = "NA";
  }
  return json_name_map;
}
}  // namespace ascend
std::string GetPyTeVersion() {
  return "";
}
}  // namespace kernel
}  // namespace mindspore
