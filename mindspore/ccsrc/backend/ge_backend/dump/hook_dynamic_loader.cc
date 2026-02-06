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

#include "backend/ge_backend/dump/hook_dynamic_loader.h"
#include <sys/stat.h>
#include <cstdlib>
#include <cstring>
#include "utils/log_adapter.h"

namespace {

// Utility function to check if a file path is valid
bool IsValidPath(const std::string &path) {
  struct stat fileStat;
  if (stat(path.c_str(), &fileStat) != 0) {
    MS_LOG(ERROR) << "File does not exist or cannot be accessed: " << path;
    return false;
  }

  if (S_ISLNK(fileStat.st_mode)) {
    MS_LOG(ERROR) << "File is a symbolic link, which is not allowed: " << path;
    return false;
  }

  if (!S_ISREG(fileStat.st_mode)) {
    MS_LOG(ERROR) << "File is not a regular file: " << path;
    return false;
  }

  if (path.substr(path.find_last_of(".")) != ".so") {
    MS_LOG(ERROR) << "File is not a .so file: " << path;
    return false;
  }

  return true;
}

}  // namespace
namespace mindspore {
namespace datadump {
HookDynamicLoader &HookDynamicLoader::GetInstance() {
  static HookDynamicLoader instance;
  return instance;
}

bool HookDynamicLoader::loadFunction(void *handle, const std::string &functionName) {
  void *func = dlsym(handle, functionName.c_str());
  if (!func) {
    MS_LOG(WARNING) << "Could not load function: " << functionName << ", error: " << dlerror();
    return false;
  }
  funcMap_[functionName] = func;
  return true;
}

bool HookDynamicLoader::validateLibraryPath(const std::string &libPath) {
  std::string realPath;
  char *tempPath = realpath(libPath.c_str(), nullptr);
  if (tempPath != nullptr) {
    realPath = tempPath;
    free(tempPath);
  }
  if (realPath.empty()) {
    MS_LOG(WARNING) << "Failed to resolve realpath for the library: " << libPath;
    return false;
  }

  bool isValid = IsValidPath(realPath);
  return isValid;
}

bool HookDynamicLoader::LoadLibrary() {
  const char *libPath = std::getenv("HOOK_TOOL_PATH");
  if (!libPath) {
    MS_LOG(WARNING) << "HOOK_TOOL_PATH is not set!";
    return false;
  }

  std::string resolvedLibPath(libPath);
  if (!validateLibraryPath(resolvedLibPath)) {
    MS_LOG(WARNING) << "Library path validation failed.";
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (handle_) {
    MS_LOG(WARNING) << "Hook library already loaded!";
    return false;
  }

  handle_ = dlopen(resolvedLibPath.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (!handle_) {
    MS_LOG(WARNING) << "Failed to load Hook library: " << dlerror();
    return false;
  }

  for (const auto &functionName : functionList_) {
    if (!loadFunction(handle_, functionName)) {
      MS_LOG(WARNING) << "Failed to load function: " << functionName;
      dlclose(handle_);
      handle_ = nullptr;
      return false;
    }
  }

  MS_LOG(INFO) << "Hook library loaded successfully.";
  return true;
}

void *HookDynamicLoader::GetHooker(const std::string &funcName) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = funcMap_.find(funcName);
  if (iter == funcMap_.end()) {
    MS_LOG(WARNING) << "Function not found: " << funcName;
    return nullptr;
  }
  return iter->second;
}
}  // namespace datadump
}  // namespace mindspore
