/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/internal/internal_kernel_build.h"

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
static std::shared_ptr<KernelPlugin> k_internal_kernel_plugin_ptr = nullptr;
static bool k_is_plugin_init = false;
std::shared_ptr<KernelPlugin> GetKernelPLugin() {
  if (k_is_plugin_init) {
    return k_internal_kernel_plugin_ptr;
  }

  // tryp load so file
  std::string cur_so_path;
  auto ret = plugin_loader::PluginLoader::GetPluginPath(&cur_so_path);
  MS_LOG(INFO) << "get so path " << cur_so_path << " return " << ret;
  auto targe_so_path = cur_so_path + "/ascend/" + "libmindspore_internal_kernels.so";

  std::stringstream dlopen_error_msg;
  std::map<std::string, void *> plugin_maps;
  ret = plugin_loader::PluginLoader::LoadDynamicLib(targe_so_path, &plugin_maps, &dlopen_error_msg);
  if (!ret) {
    MS_LOG(ERROR) << "load so failed " << dlopen_error_msg.str();
  }

  // create plugin object
  k_internal_kernel_plugin_ptr = Factory<KernelPlugin>::Instance().Create("InternalKernelPlugin");
  k_is_plugin_init = true;
  if (k_internal_kernel_plugin_ptr != nullptr) {
    k_internal_kernel_plugin_ptr->InitInternalLog();
  }
  return k_internal_kernel_plugin_ptr;
}

KernelModPtr InternalKernelBuild(const AnfNodePtr &anf_node) {
  auto internal_kernel_plugin_ptr = GetKernelPLugin();
  if (internal_kernel_plugin_ptr == nullptr) {
    return nullptr;
  }
  return internal_kernel_plugin_ptr->BuildKernel(anf_node);
}

bool IsRegisteredInternalKernel(const AnfNodePtr &anf_node) {
  auto internal_kernel_plugin_ptr = GetKernelPLugin();
  if (internal_kernel_plugin_ptr == nullptr) {
    return false;
  }
  return internal_kernel_plugin_ptr->IsRegisteredKernel(anf_node);
}

void GetValidKernelBuildInfoWithInternalFormat(const AnfNodePtr &node, std::vector<std::string> *input_formats,
                                               std::vector<std::string> *output_formats) {
  auto internal_kernel_plugin_ptr = GetKernelPLugin();
  if (internal_kernel_plugin_ptr == nullptr) {
    return;
  }
  return internal_kernel_plugin_ptr->GetValidKernelBuildInfoWithInternalFormat(node, input_formats, output_formats);
}

bool IsEnableInternalNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!context_ptr->IsEnableInferBoost()) {
    return false;
  }

  std::string op_name = common::AnfAlgo::GetCNodeName(node);
  if (op_name == "SplitWithSize") {
    static const auto kSplitOutNum2 = 2;
    static const auto kSplitOutNum3 = 3;
    auto out_num = AnfUtils::GetOutputTensorNum(node);
    if (out_num != kSplitOutNum2 && out_num != kSplitOutNum3) {
      MS_LOG(INFO) << "Split only support 2 or 3 outputs, but got: " << out_num;
      return false;
    }
  }

  std::string disable_op_env = common::GetEnv("MS_DISABLE_INTERNAL_KERNELS_LIST");
  std::set<std::string> disable_op_list;
  common::SplitString(disable_op_env, ',', &disable_op_list);
  bool disable_internal_op =
    (std::find(disable_op_list.begin(), disable_op_list.end(), op_name) != disable_op_list.end());
  if (disable_internal_op) {
    return false;
  }

  return IsRegisteredInternalKernel(node);
}
}  // namespace kernel
}  // namespace mindspore
