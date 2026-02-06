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

#include "kernel/ascend/symmetric_memory/symmetric_memory_kernel_build.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>

#include "kernel/ascend/kernel_plugin.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/utils/anfalgo.h"

namespace mindspore {
namespace kernel {
static std::shared_ptr<KernelPlugin> k_symmetric_memory_kernel_plugin_ptr = nullptr;
static bool k_is_plugin_init = false;
std::shared_ptr<KernelPlugin> GetSymmetricMemoryKernelPLugin() {
  if (k_is_plugin_init) {
    return k_symmetric_memory_kernel_plugin_ptr;
  }

  // try to load so file
  std::string cur_so_path;
  auto ret = plugin_loader::PluginLoader::GetPluginPath(&cur_so_path);
  MS_LOG(INFO) << "get so path " << cur_so_path << " return " << ret;
  auto targe_so_path = cur_so_path + "/ascend/" + "libmindspore_symmetric_memory_kernels.so";

  std::stringstream dlopen_error_msg;
  std::map<std::string, void *> plugin_maps;
  ret = plugin_loader::PluginLoader::LoadDynamicLib(targe_so_path, &plugin_maps, &dlopen_error_msg);
  if (!ret) {
    MS_LOG(ERROR) << "load so failed " << dlopen_error_msg.str();
  }

  // create plugin object
  k_symmetric_memory_kernel_plugin_ptr = Factory<KernelPlugin>::Instance().Create("SymmetricMemoryKernelPlugin");
  k_is_plugin_init = true;
  if (k_symmetric_memory_kernel_plugin_ptr != nullptr) {
    k_symmetric_memory_kernel_plugin_ptr->InitInternalLog();
  }
  return k_symmetric_memory_kernel_plugin_ptr;
}

KernelModPtr SymmetricMemoryKernelBuild(const AnfNodePtr &anf_node) {
  auto symmetric_memory_kernel_plugin_ptr = GetSymmetricMemoryKernelPLugin();
  if (symmetric_memory_kernel_plugin_ptr == nullptr) {
    return nullptr;
  }
  return symmetric_memory_kernel_plugin_ptr->BuildKernel(anf_node);
}

void GetValidKernelBuildInfoWithSymmetricMemoryFormat(const AnfNodePtr &node, std::vector<std::string> *input_formats,
                                                      std::vector<std::string> *output_formats) {
  auto symmetric_memory_kernel_plugin_ptr = GetSymmetricMemoryKernelPLugin();
  if (symmetric_memory_kernel_plugin_ptr == nullptr) {
    return;
  }
  return symmetric_memory_kernel_plugin_ptr->GetValidKernelBuildInfoWithInternalFormat(node, input_formats,
                                                                                       output_formats);
}

bool IsEnableSymmetricMemoryNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto op_name = common::AnfAlgo::GetCNodeName(node);
  if (enableSymmetricMemoryNodeList.find(op_name) == enableSymmetricMemoryNodeList.end()) {
    return false;
  }
  auto symmetric_memory_kernel_plugin_ptr = GetSymmetricMemoryKernelPLugin();
  if (symmetric_memory_kernel_plugin_ptr == nullptr) {
    return false;
  }
  return symmetric_memory_kernel_plugin_ptr->IsRegisteredKernel(node);
}
}  // namespace kernel
}  // namespace mindspore
