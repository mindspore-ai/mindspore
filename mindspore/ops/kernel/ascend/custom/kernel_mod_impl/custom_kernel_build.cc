/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/custom/kernel_mod_impl/custom_kernel_build.h"
#include "kernel/ascend/kernel_plugin.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "kernel/ascend/custom/kernel_mod_impl/custom_kernel_factory.h"
#include "include/utils/anfalgo.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace kernel {
static std::shared_ptr<KernelPlugin> k_custom_kernel_plugin_ptr = nullptr;
static bool k_is_custom_plugin_init = false;
std::shared_ptr<KernelPlugin> GetCustomKernelPlugin() {
  if (k_is_custom_plugin_init) {
    return k_custom_kernel_plugin_ptr;
  }

  // create plugin object
  k_custom_kernel_plugin_ptr = Factory<KernelPlugin>::Instance().Create("CustomKernelPlugin");
  k_is_custom_plugin_init = true;

  return k_custom_kernel_plugin_ptr;
}

KernelModPtr CustomKernelBuild(const AnfNodePtr &anf_node) {
  k_custom_kernel_plugin_ptr = GetCustomKernelPlugin();
  if (k_custom_kernel_plugin_ptr == nullptr) {
    return nullptr;
  }
  return k_custom_kernel_plugin_ptr->BuildKernel(anf_node);
}

bool IsRegisteredCustomKernel(const AnfNodePtr &anf_node) {
  k_custom_kernel_plugin_ptr = GetCustomKernelPlugin();
  if (k_custom_kernel_plugin_ptr == nullptr) {
    return false;
  }
  return k_custom_kernel_plugin_ptr->IsRegisteredKernel(anf_node);
}

bool IsEnableCustomNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return IsRegisteredCustomKernel(node);
}
}  // namespace kernel
}  // namespace mindspore
