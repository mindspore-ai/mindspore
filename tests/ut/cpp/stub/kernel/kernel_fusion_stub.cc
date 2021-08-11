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
#include "backend/kernel_compiler/kernel_fusion.h"
#include "backend/kernel_compiler/tbe/tbe_kernel_mod.h"
#include "backend/kernel_compiler/tbe/ascend_kernel_compile.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
std::map<int64_t, KernelModPtr> KernelFusion(const std::vector<FusionScopeInfo> &fusion_scopes) {
  std::map<int64_t, KernelModPtr> kernel_mod_ret;
  for (const auto &fusion_scope_iter : fusion_scopes) {
    kernel_mod_ret[fusion_scope_iter.scope_id] = std::make_shared<TbeKernelMod>(nullptr);
  }
  return kernel_mod_ret;
}
namespace ascend {
std::string AscendKernelCompileManager::AscendOpSelectFormat(const AnfNodePtr &node) { return std::string(); }
bool AscendKernelCompileManager::AscendOpCheckSupported(const AnfNodePtr &node) { return true; }
AscendKernelCompileManager::~AscendKernelCompileManager() {}
bool AscendKernelCompileManager::tbe_init_flag_ = true;

void AscendKernelCompileManager::TbeInitialize() {}
// pre build
void AscendKernelCompileManager::AscendPreBuild(const std::shared_ptr<session::KernelGraph> &kernel_graph) {}
// single op compile
bool AscendKernelCompileManager::AscendSingleOpCompile(const std::vector<AnfNodePtr> &anf_nodes) { return true; }
// fusion op compile
KernelModMap AscendKernelCompileManager::AscendFusionOpCompile(const std::vector<FusionScopeInfo> &fusion_scopes) {
  std::map<int64_t, KernelModPtr> kernel_mod_ret;
  for (const auto &fusion_scope_iter : fusion_scopes) {
    kernel_mod_ret[fusion_scope_iter.scope_id] = std::make_shared<TbeKernelMod>(nullptr);
  }
  return kernel_mod_ret;
}
void AscendKernelCompileManager::ResetOldTask() {}
}  // namespace ascend
}  // namespace kernel
}  // namespace mindspore
