/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "device/ascend/profiling/profiling_engine_impl.h"
#include "utils/log_adapter.h"
#include "device/ascend/profiling/plugin_impl.h"

namespace mindspore {
namespace device {
namespace ascend {
PluginIntf *ProfilingEngineImpl::CreatePlugin() {
  MS_LOG(INFO) << "Create Plugin.";
  return new (std::nothrow) PluginImpl("Framework");
}

int ProfilingEngineImpl::ReleasePlugin(PluginIntf *plugin) {
  if (plugin != nullptr) {
    delete plugin;
    plugin = nullptr;
  }
  return 0;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
