/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef HOOK_DYNAMIC_LOADER_H
#define HOOK_DYNAMIC_LOADER_H

#include <dlfcn.h>
#include <string>
#include <vector>
#include <map>
#include <mutex>

namespace mindspore {
namespace datadump {
constexpr auto kHookBegin = "MS_DbgOnStepBegin";
constexpr auto kHookEnd = "MS_DbgOnStepEnd";

class HookDynamicLoader {
 public:
  static HookDynamicLoader &GetInstance();

  HookDynamicLoader(const HookDynamicLoader &) = delete;
  HookDynamicLoader &operator=(const HookDynamicLoader &) = delete;

  bool LoadLibrary();
  void *GetHooker(const std::string &funcName);

 private:
  // Helper functions
  bool loadFunction(void *handle, const std::string &functionName);
  bool validateLibraryPath(const std::string &libPath);

  HookDynamicLoader() = default;

  void *handle_ = nullptr;
  std::vector<std::string> functionList_ = {kHookBegin, kHookEnd};
  std::map<std::string, void *> funcMap_;
  std::mutex mutex_;
};
}  // namespace datadump
}  // namespace mindspore

#endif  // HOOK_DYNAMIC_LOADER_H
