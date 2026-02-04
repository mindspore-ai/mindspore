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
#include "kernel/ascend/atb/kernel_mod_impl/atb_kernel_build.h"
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "include/utils/anfalgo.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"
#include "include/op_def.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "kernel/ascend/kernel_plugin.h"
#include "include/utils/utils.h"

namespace mindspore {
namespace kernel {

static std::shared_ptr<KernelPlugin> k_atb_kernel_plugin_ptr = nullptr;
static bool k_is_plugin_init = false;

std::shared_ptr<KernelPlugin> GetAtbKernelPLugin() {
  if (k_is_plugin_init) {
    return k_atb_kernel_plugin_ptr;
  }

  // try load so file
  std::string cur_so_path;
  auto ret = plugin_loader::PluginLoader::GetPluginPath(&cur_so_path);
  MS_LOG(INFO) << "Get so path " << cur_so_path << " return " << ret;
  auto targe_so_path = cur_so_path + "/ascend/" + "libmindspore_atb_kernels.so";

  std::stringstream dlopen_error_msg;
  std::map<std::string, void *> plugin_maps;
  ret = plugin_loader::PluginLoader::LoadDynamicLib(targe_so_path, &plugin_maps, &dlopen_error_msg);
  if (!ret) {
    MS_LOG(INFO) << "Load so failed " << dlopen_error_msg.str()
                 << ", you can enable ATB by install the nnal package and source the set_env.sh in nnal.";
    k_atb_kernel_plugin_ptr = nullptr;
    k_is_plugin_init = true;
    return nullptr;
  }

  // create plugin object
  k_atb_kernel_plugin_ptr = Factory<KernelPlugin>::Instance().Create("AtbKernelPlugin");
  k_is_plugin_init = true;
  return k_atb_kernel_plugin_ptr;
}

KernelModPtr AtbKernelBuild(const AnfNodePtr &anf_node) {
  auto atb_kernel_plugin = GetAtbKernelPLugin();
  if (atb_kernel_plugin == nullptr) {
    return nullptr;
  }
  return atb_kernel_plugin->BuildKernel(anf_node);
}

bool IsRegisteredAtbKernel(const AnfNodePtr &anf_node) {
  auto atb_kernel_plugin = GetAtbKernelPLugin();
  if (atb_kernel_plugin == nullptr) {
    return false;
  }
  return atb_kernel_plugin->IsRegisteredKernel(anf_node);
}

bool IsEnableAtb(const KernelGraphPtr &kernel_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(node);
#ifdef ENABLE_ATB
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->IsEnableInferBoost()) {
    MS_LOG(INFO) << "Infer boost is enable, set ATB unenable.";
    return false;
  }
  static bool special_format = GetFormatMode() == "0";
  if (special_format) {
    return false;
  }
  if (kernel_graph->is_from_single_op()) {
    return false;
  }

  return IsRegisteredAtbKernel(node);
#else
  MS_LOG(INFO) << "Build mindspore without nnal package, ATB is not enable.";
  return false;
#endif
}
}  // namespace kernel
}  // namespace mindspore
