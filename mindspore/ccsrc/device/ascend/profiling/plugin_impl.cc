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
#include "device/ascend/profiling/plugin_impl.h"
#include <string>
#include "utils/log_adapter.h"
using std::string;

namespace mindspore {
namespace device {
namespace ascend {
Reporter *PluginImpl::reporter_ = nullptr;

PluginImpl::PluginImpl(const std::string &module) : module_(module) { MS_LOG(INFO) << "Create PluginImpl."; }

int PluginImpl::Init(const Reporter *reporter) {
  MS_LOG(INFO) << "PluginImpl init";
  MS_EXCEPTION_IF_NULL(reporter);
  reporter_ = const_cast<Reporter *>(reporter);
  return 0;
}

int PluginImpl::UnInit() {
  MS_LOG(INFO) << " PluginImpl Uninit ";
  reporter_ = nullptr;
  return 0;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
