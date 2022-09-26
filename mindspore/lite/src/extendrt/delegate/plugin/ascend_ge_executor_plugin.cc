/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "extendrt/delegate/plugin/ascend_ge_executor_plugin.h"
#include <string>
#include "utils/log_adapter.h"
#if !defined(_WIN32)
#include "extendrt/cxx_api/dlutils.h"
#endif

namespace mindspore::lite {
namespace {
constexpr auto kAscendGePluginSoName = "ascend_ge_plugin.so";
}  // namespace
AscendGeExecutorPlugin::AscendGeExecutorPlugin() = default;
AscendGeExecutorPlugin::~AscendGeExecutorPlugin() {
#if !defined(_WIN32)
  MS_LOG(DEBUG) << "~AscendGeExecutorPlugin() begin.";
  DLSoClose(handle_);
  is_registered_ = false;
  MS_LOG(DEBUG) << "~AscendGeExecutorPlugin() end.";
#endif
}

AscendGeExecutorPlugin &AscendGeExecutorPlugin::GetInstance() {
  static AscendGeExecutorPlugin instance;
  return instance;
}

bool AscendGeExecutorPlugin::Register() {
#if !defined(_WIN32)
  if (is_registered_) {
    return true;
  }
  std::string plugin_path;
  auto ret = DLSoPath("libmindspore-lite.so", kAscendGePluginSoName, &plugin_path);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Get real path of " << kAscendGePluginSoName << " failed, ret = " << ret;
    return false;
  }
  MS_LOG(INFO) << "Find tensorrt plugin so success, path = " << plugin_path;
  void *function = nullptr;
  ret = DLSoOpen(plugin_path, "", &handle_, &function);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "DLSoOpen failed, so path: " << plugin_path;
    return false;
  }
  is_registered_ = true;
  MS_LOG(INFO) << "Register tensorrt plugin success.";
#endif
  return true;
}
}  // namespace mindspore::lite
