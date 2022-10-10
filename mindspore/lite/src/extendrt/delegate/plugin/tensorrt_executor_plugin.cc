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
#include "extendrt/delegate/plugin/tensorrt_executor_plugin.h"
#include <string>
#include "utils/log_adapter.h"
#if !defined(_WIN32)
#include "extendrt/cxx_api/dlutils.h"
#endif

namespace mindspore::lite {
namespace {
constexpr auto kTensorRtPluginSoName = "libtensorrt_plugin.so";
constexpr auto kFunCreateTRTPluginImp = "CreateTensorRTPluginImpl";
}  // namespace
TensorRTExecutorPlugin::TensorRTExecutorPlugin() = default;
TensorRTExecutorPlugin::~TensorRTExecutorPlugin() {
#if !defined(_WIN32)
  MS_LOG(DEBUG) << "~TensorRTExecutorPlugin() begin.";
  DLSoClose(handle_);
  MS_LOG(DEBUG) << "~TensorRTExecutorPlugin() end.";
#endif
}

TensorRTExecutorPlugin &TensorRTExecutorPlugin::GetInstance() {
  static TensorRTExecutorPlugin instance;
  return instance;
}

bool TensorRTExecutorPlugin::Register() {
#if !defined(_WIN32)
  if (is_registered_) {
    return true;
  }
  std::string plugin_path;
  auto ret = DLSoPath("libmindspore-lite.so", kTensorRtPluginSoName, &plugin_path);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Get real path of " << kTensorRtPluginSoName << " failed.";
    return false;
  }
  MS_LOG(INFO) << "Find tensorrt plugin so success, path = " << plugin_path;
  void *function = nullptr;
  ret = DLSoOpen(plugin_path, kFunCreateTRTPluginImp, &handle_, &function);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "DLSoOpen failed, so path: " << plugin_path;
    return false;
  }
  auto create_kernel_func = reinterpret_cast<mindspore::lite::TensorRTExecutorPluginImplBase *(*)(void)>(function);
  if (create_kernel_func == nullptr) {
    MS_LOG(ERROR) << "Cast " << kFunCreateTRTPluginImp << " failed.";
    return false;
  }
  auto plugin_impl = create_kernel_func();
  if (plugin_impl == nullptr) {
    MS_LOG(ERROR) << "Create custom TensorRT kernel failed.";
    return false;
  }
  group_size_ = plugin_impl->GetGPUGroupSize();
  rank_id_ = plugin_impl->GetRankID();
  is_registered_ = true;
  MS_LOG(INFO) << "Register tensorrt plugin success.";
#endif
  return true;
}

int TensorRTExecutorPlugin::GetGPUGroupSize() {
#ifdef SUPPORT_TENSORRT
  if (!is_registered_) {
    Register();
  }
#endif
  return group_size_;
}

int TensorRTExecutorPlugin::GetRankID() {
#ifdef SUPPORT_TENSORRT
  if (!is_registered_) {
    Register();
  }
#endif
  return rank_id_;
}
}  // namespace mindspore::lite
