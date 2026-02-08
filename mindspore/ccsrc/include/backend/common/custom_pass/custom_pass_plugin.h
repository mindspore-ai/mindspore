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

#ifndef MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_CUSTOM_PASS_CUSTOM_PASS_PLUGIN_H_
#define MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_CUSTOM_PASS_CUSTOM_PASS_PLUGIN_H_

#include <string>
#include <vector>
#include "include/backend/common/pass_manager/pass.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace opt {

class BACKEND_COMMON_EXPORT CustomPassPlugin {
 public:
  virtual ~CustomPassPlugin() = default;

  virtual std::string GetPluginName() const = 0;

  // Get available pass names
  virtual std::vector<std::string> GetAvailablePassNames() const = 0;

  // Create pass by name
  virtual std::shared_ptr<Pass> CreatePass(const std::string &pass_name) const = 0;

  // Default implementation: always enabled, override if needed
  virtual bool IsEnabled() const { return true; }
};
}  // namespace opt
}  // namespace mindspore

#define EXPORT_CUSTOM_PASS_PLUGIN(PluginClass)                                                          \
  extern "C" {                                                                                          \
  BACKEND_COMMON_EXPORT mindspore::opt::CustomPassPlugin *CreatePlugin() { return new PluginClass(); }  \
  BACKEND_COMMON_EXPORT void DestroyPlugin(mindspore::opt::CustomPassPlugin *plugin) { delete plugin; } \
  BACKEND_COMMON_EXPORT const char *GetPluginName() { return #PluginClass; }                            \
  }

#endif  // MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_CUSTOM_PASS_CUSTOM_PASS_PLUGIN_H_
