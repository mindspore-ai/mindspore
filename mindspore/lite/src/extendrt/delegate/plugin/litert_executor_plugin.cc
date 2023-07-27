/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include <string>

#include "extendrt/delegate/plugin/litert_executor_plugin.h"
#include "utils/log_adapter.h"
#if !defined(_WIN32)
#include "extendrt/cxx_api/dlutils.h"
#endif

namespace mindspore::infer {
namespace {
constexpr auto kLiteRtPluginSoName = "libmsplugin-ge-litert.so";
constexpr auto kFunCreateLiteRTPluginImp = "CreateLiteRTPluginImpl";
}  // namespace
LiteRTExecutorPlugin::LiteRTExecutorPlugin() = default;
LiteRTExecutorPlugin::~LiteRTExecutorPlugin() {
#if !defined(_WIN32)
  MS_LOG(DEBUG) << "~LiteRTExecutorPlugin() begin.";
  DLSoClose(handle_);
  MS_LOG(DEBUG) << "~LiteRTExecutorPlugin() end.";
#endif
}

LiteRTExecutorPlugin &LiteRTExecutorPlugin::GetInstance() {
  static LiteRTExecutorPlugin instance;
  return instance;
}

bool LiteRTExecutorPlugin::Register() {
#if !defined(_WIN32)
  if (is_registered_) {
    return true;
  }
  std::string plugin_path;
  auto ret = DLSoPath({"libmindspore-lite.so", "_c_lite"}, kLiteRtPluginSoName, &plugin_path);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Get real path of " << kLiteRtPluginSoName << " failed.";
    return false;
  }
  MS_LOG(INFO) << "Find litert plugin so success, path = " << plugin_path;
  void *function = nullptr;
  ret = DLSoOpen(plugin_path, kFunCreateLiteRTPluginImp, &handle_, &function);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "DLSoOpen failed, so path: " << plugin_path;
    return false;
  }
  auto create_kernel_func = reinterpret_cast<mindspore::infer::LiteRTExecutorPluginImplBase *(*)(void)>(function);
  if (create_kernel_func == nullptr) {
    MS_LOG(ERROR) << "Cast " << kFunCreateLiteRTPluginImp << " failed.";
    return false;
  }
  auto plugin_impl = create_kernel_func();
  if (plugin_impl == nullptr) {
    MS_LOG(ERROR) << "Create custom litert kernel failed.";
    return false;
  }
  delete plugin_impl;
  is_registered_ = true;
  MS_LOG(INFO) << "Register litert plugin success.";
#endif
  return true;
}
}  // namespace mindspore::infer
