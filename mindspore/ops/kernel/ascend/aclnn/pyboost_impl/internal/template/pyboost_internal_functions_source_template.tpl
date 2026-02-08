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
#include "kernel/ascend/aclnn/pyboost_impl/internal/functions/functions.h"

#include <string>
#include <utility>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>

#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"

namespace mindspore {
namespace kernel {
static bool is_plugin_loaded = false;
void LoadPlugin() {
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
  is_plugin_loaded = true;
}
${func_list}
}  // namespace kernel
}  // namespace mindspore
