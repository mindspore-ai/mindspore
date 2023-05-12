/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
constexpr auto kAscendGePluginSoName = "libascend_ge_plugin.so";
constexpr auto kAscendGePluginPySoName = "libascend_ge_plugin_python.so";
constexpr auto kFunCreateAscendGePluginImpl = "CreateAscendGeExecutorPluginImpl";
}  // namespace
AscendGeExecutorPlugin::AscendGeExecutorPlugin() = default;
AscendGeExecutorPlugin::~AscendGeExecutorPlugin() {
#if !defined(_WIN32)
  MS_LOG(DEBUG) << "~AscendGeExecutorPlugin() begin.";
  if (ge_plugin_impl_ != nullptr) {
    delete ge_plugin_impl_;
    ge_plugin_impl_ = nullptr;
  }
  DLSoClose(handle_);
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
  auto ret = DLSoPath({"libmindspore-lite.so", "_c_lite"}, kAscendGePluginSoName, &plugin_path_);
  if (ret != kSuccess) {
    MS_LOG(INFO) << "Get real path of " << kAscendGePluginSoName << " failed, then Get real path of "
                 << kAscendGePluginSoName;
    ret = DLSoPath({"libmindspore-extendrt_python.so", "_c_lite"}, kAscendGePluginPySoName, &plugin_path_);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Get real path of " << kAscendGePluginPySoName << " failed.";
      return false;
    }
  }
  MS_LOG(INFO) << "Find tensorrt plugin so success, path = " << plugin_path_;
  void *function = nullptr;
  ret = DLSoOpen(plugin_path_, kFunCreateAscendGePluginImpl, &handle_, &function);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "DLSoOpen failed, so path: " << plugin_path_;
    return false;
  }
  auto create_plugin_impl_func = reinterpret_cast<AscendGeExecutorPluginImplBase *(*)(void)>(function);
  if (create_plugin_impl_func == nullptr) {
    MS_LOG(ERROR) << "Cast " << kFunCreateAscendGePluginImpl << " failed.";
    return false;
  }
  ge_plugin_impl_ = create_plugin_impl_func();
  if (ge_plugin_impl_ == nullptr) {
    MS_LOG(ERROR) << "Create Ascend ge plugin implement failed.";
    return false;
  }
  is_registered_ = true;
  MS_LOG(INFO) << "Register Ascend ge plugin success.";
#endif
  return true;
}

Status AscendGeExecutorPlugin::InitializeGeContext(const std::shared_ptr<Context> &context,
                                                   const ConfigInfos &config_info) {
  Status ret = kSuccess;
#if !defined(_WIN32)
  if (!is_registered_ || ge_plugin_impl_ == nullptr) {
    MS_LOG(ERROR) << "The Ascend ge executor is not registered.";
    return kLiteError;
  }
  ret = ge_plugin_impl_->AscendGeDeviceContextInitialize(context, config_info);
#endif
  return ret;
}

void AscendGeExecutorPlugin::DestroyGeContext() {
#if !defined(_WIN32)
  if (!is_registered_ || ge_plugin_impl_ == nullptr) {
    MS_LOG(ERROR) << "The Ascend ge executor is not registered.";
    return;
  }
  ge_plugin_impl_->AscendGeDeviceContextDestroy();
#endif
}

void AscendGeExecutorPlugin::AdaptGraph(FuncGraphPtr graph) {
  MS_ASSERT(graph != nullptr);
#if !defined(_WIN32)
  if (!is_registered_ || ge_plugin_impl_ == nullptr) {
    MS_LOG(ERROR) << "The Ascend ge executor is not registered.";
    return;
  }
  (void)ge_plugin_impl_->AdaptGraph(graph);
#endif
}
}  // namespace mindspore::lite
