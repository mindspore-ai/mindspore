/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/akg/akg_gpu_kernel_build.h"
#include <memory>
#include <string>
#include "kernel/common_utils.h"
#include "plugin/device/gpu/kernel/akg/akg_gpu_kernel_mod.h"
#include "utils/ms_utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"

namespace mindspore {
namespace kernel {
constexpr int32_t ARGS_SIZE = 1;

void AkgGpuKernelBuilder::AkgSetKernelMod(const KernelPackPtr &kernel_pack,
                                          const AkgKernelJsonGenerator &json_generator, const AnfNodePtr &anf_node) {
  const auto &flags = graphkernel::GraphKernelFlags::GetInstance();
  auto kernel_mod_ptr = flags.enable_debug_mode ? std::make_shared<AkgGpuKernelModDebug>(kernel_pack)
                                                : std::make_shared<AkgGpuKernelMod>(kernel_pack);
  auto kernel_json_info = kernel_pack->kernel_json_info();
  kernel_mod_ptr->SetInputSizeList(json_generator.input_size_list());
  kernel_mod_ptr->SetOutputSizeList(json_generator.output_size_list());
  kernel_mod_ptr->SetWorkspaceSizeList(kernel_json_info.workspaces);
  AnfAlgo::SetKernelMod(kernel_mod_ptr, anf_node.get());
}

void AkgGpuKernelBuilder::AkgSaveJsonInfo(const string &kernel_name, const string &kernel_json) {
  kernel::SaveJsonInfo(kernel_name, kernel_json, kernel::KernelMeta::GetInstance()->kernel_meta_path());
}
}  // namespace kernel
}  // namespace mindspore
