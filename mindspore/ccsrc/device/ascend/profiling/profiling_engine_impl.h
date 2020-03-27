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
#ifndef MINDSPORE_CCSRC_DEVICE_ASCEND_PROFILING_PROFILING_ENGINE_IMPL_H_
#define MINDSPORE_CCSRC_DEVICE_ASCEND_PROFILING_PROFILING_ENGINE_IMPL_H_

#include "./prof_engine.h"

using Msprof::Engine::EngineIntf;
using Msprof::Engine::PluginIntf;

namespace mindspore {
namespace device {
namespace ascend {
class ProfilingEngineImpl : public EngineIntf {
 public:
  ProfilingEngineImpl() = default;
  ~ProfilingEngineImpl() override = default;

  PluginIntf* CreatePlugin() override;
  int ReleasePlugin(PluginIntf* plugin) override;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEVICE_ASCEND_PROFILING_PROFILING_ENGINE_IMPL_H_
