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

#include <memory>
#include <string>
#include <vector>
#include "kernel/common_utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/cpu/kernel/akg/akg_cpu_kernel_build.h"
#include "plugin/device/cpu/kernel/bisheng/bisheng_cpu_kernel_build.h"
#include "plugin/device/cpu/kernel/bisheng/bisheng_cpu_kernel_mod.h"

namespace mindspore {
namespace kernel {
void BishengCpuKernelBuilder::BishengSetKernelMod(const string &kernel_name,
                                                  const AkgKernelJsonGenerator &json_generator,
                                                  const AnfNodePtr &anf_node) {
  auto kernel_mod_ptr = std::make_shared<BishengCpuKernelMod>(kernel_name);
  kernel_mod_ptr->SetInputSizeList(json_generator.input_size_list());
  kernel_mod_ptr->SetOutputSizeList(json_generator.output_size_list());

  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto args = kernel::AbstractArgsFromCNode(cnode, true);
  bool is_dynamic_kernel =
    std::any_of(args.inputs.begin(), args.inputs.end(), [](KernelTensorPtr item) { return item->IsDynamicShape(); }) ||
    std::any_of(args.outputs.begin(), args.outputs.end(), [](KernelTensorPtr item) { return item->IsDynamicShape(); });
  kernel_mod_ptr->SetKernelDynamicStatus(is_dynamic_kernel);
  AnfAlgo::SetKernelMod(kernel_mod_ptr, anf_node.get());
}

void BishengCpuKernelBuilder::AkgSaveJsonInfo(const string &kernel_name, const string &kernel_json) {
  auto config_path = GetCompilerCachePath();
  auto kernel_meta_path = config_path + std::string(kBishengKernelMeta);
  kernel::SaveJsonInfo(kernel_name, kernel_json, kernel_meta_path);
}

bool BishengCpuKernelBuilder::ParallelBuild(const std::vector<JsonNodePair> &json_and_node) {
  for (const auto &[json_generator, anf_node] : json_and_node) {
    auto kernel_name = json_generator.kernel_name();
    auto kernel_json = json_generator.kernel_json_str();
    AkgSaveJsonInfo(kernel_name, kernel_json);
    BishengSetKernelMod(kernel_name, json_generator, anf_node);
    MS_LOG(DEBUG) << "Bisheng set " << kernel_name << " kernel successfully!";
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
