/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "runtime/hardware/device_context_manager.h"
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif
#include <dirent.h>
#include <algorithm>
#include <string>
#include <fstream>
#include "utils/ms_context.h"
#include "utils/dlopen_macro.h"
#include "utils/os.h"

namespace mindspore {
namespace plugin_loader {
void PluginLoader::LoadDynamicLib(const std::string &plugin_file, std::map<std::string, void *> *all_handles) {
  MS_EXCEPTION_IF_NULL(all_handles);
  void *handle = nullptr;
  std::string err_msg;
#ifndef _WIN32
  if (plugin_file.find("libmindspore_") == std::string::npos) {
    return;
  }
#else
  if (plugin_file.find("mindspore_") == std::string::npos) {
    return;
  }
#endif
  auto so_name = GetDynamicLibName(plugin_file);
#if defined(_WIN32) || defined(_WIN64)
  handle = LoadLibraryEx(plugin_file.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
  err_msg = std::to_string(GetLastError());
#else
  handle = dlopen(plugin_file.c_str(), RTLD_NOW | RTLD_LOCAL);
  err_msg = GetDlErrorMsg();
#endif
  if (handle == nullptr) {
    MS_LOG(INFO) << "Load dynamic lib: " << so_name << " failed. " << err_msg;
    return;
  }
  (*all_handles)[so_name] = handle;
}

void PluginLoader::CloseDynamicLib(const std::string &dl_name, void *handle) {
#if defined(_WIN32) || defined(_WIN64)
  if (!FreeLibrary(static_cast<HMODULE>(handle))) {
    MS_LOG(EXCEPTION) << "Closing dynamic lib: " + dl_name + " handle failed. Error: " + std::to_string(GetLastError());
  }

#else
  if (dlclose(handle) != 0) {
    MS_LOG(ERROR) << "Closing dynamic lib: " << dl_name << "failed, error message: " << GetDlErrorMsg();
  }
#endif
}

std::string PluginLoader::GetDynamicLibName(const std::string &plugin_file) {
  auto p1 = plugin_file.find_last_of(PATH_SEPARATOR) + 1;
  auto target_so = plugin_file.substr(p1);
  auto pos = target_so.rfind('.');
  if (pos == std::string::npos) {
    MS_LOG(WARNING) << "Invalid plugin file " << target_so;
    return "unknown_name";
  }
  return target_so.substr(0, pos);
}

bool PluginLoader::GetPluginPath(std::string *file_path) {
  MS_EXCEPTION_IF_NULL(file_path);
  std::string cur_so_path;
#if !defined(_WIN32) && !defined(_WIN64)
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void *>(PluginLoader::GetPluginPath), &dl_info) == 0) {
    MS_LOG(INFO) << "Get dladdr error";
    return false;
  }
  cur_so_path = dl_info.dli_fname;
#else
  HMODULE hModule = nullptr;
  if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                        (LPCSTR)PluginLoader::GetPluginPath, &hModule) == 0) {
    MS_LOG(INFO) << "Get GetModuleHandleEx failed.";
    return false;
  }
  char szPath[MAX_PATH];
  if (GetModuleFileName(hModule, szPath, sizeof(szPath)) == 0) {
    MS_LOG(INFO) << "Get GetModuleHandleEx failed.";
    return false;
  }
  cur_so_path = std::string(szPath);
#endif
  auto pos = cur_so_path.find_last_of(PATH_SEPARATOR);
  if (cur_so_path.empty() || pos == std::string::npos) {
    MS_LOG(INFO) << "Current so path empty or the path [" << cur_so_path << "] is invalid.";
    return false;
  }
#ifndef _WIN32
  auto plugin_so_path = cur_so_path.substr(0, pos) + "/plugin";
#else
  auto plugin_so_path = cur_so_path.substr(0, pos);
#endif
  if (plugin_so_path.size() >= PATH_MAX) {
    MS_LOG(INFO) << "Current path [" << plugin_so_path << "] is invalid.";
    return false;
  }
  char real_path_mem[PATH_MAX] = {0};
#if defined(_WIN32) || defined(_WIN64)
  if (_fullpath(real_path_mem, common::SafeCStr(plugin_so_path), PATH_MAX) == nullptr) {
    MS_LOG(INFO) << "Plugin path is invalid: [" << plugin_so_path << "], skip!";
    return false;
  }
#else
  if (realpath(common::SafeCStr(plugin_so_path), real_path_mem) == nullptr) {
    MS_LOG(INFO) << "Plugin path is invalid: [" << plugin_so_path << "], skip!";
    return false;
  }
#endif
  *file_path = std::string(real_path_mem);
  return true;
}
}  // namespace plugin_loader

namespace device {
DeviceContextManager &DeviceContextManager::GetInstance() {
  static DeviceContextManager instance{};
  instance.LoadPlugin();
  return instance;
}

void DeviceContextManager::Register(const std::string &device_name, DeviceContextCreator &&device_context_creator) {
  if (device_context_creators_.find(device_name) == device_context_creators_.end()) {
    (void)device_context_creators_.emplace(device_name, device_context_creator);
  }
}

void DeviceContextManager::LoadPlugin() {
  if (load_init_) {
    return;
  }
  if (plugin_path_.empty() && !plugin_loader::PluginLoader::GetPluginPath(&plugin_path_)) {
    MS_LOG(INFO) << "Plugin path is invalid, skip!";
    return;
  }
#ifdef _WIN32
  auto plugin_file = plugin_path_ + "\\mindspore_gpu.dll";
  plugin_loader::PluginLoader::LoadDynamicLib(plugin_file, &plugin_maps_);
#else
  DIR *dir = opendir(plugin_path_.c_str());
  if (dir == nullptr) {
    MS_LOG(ERROR) << "Open plugin dir failed, plugin path:" << plugin_path_;
    return;
  }
  struct dirent *entry;
  while ((entry = readdir(dir)) != nullptr) {
    auto plugin_file = plugin_path_ + PATH_SEPARATOR + entry->d_name;
    plugin_loader::PluginLoader::LoadDynamicLib(plugin_file, &plugin_maps_);
  }
  (void)closedir(dir);
#endif
  load_init_ = true;
}

void DeviceContextManager::UnloadPlugin() {
  if (plugin_maps_.empty()) {
    return;
  }
  auto iter = plugin_maps_.begin();
  while (iter != plugin_maps_.end()) {
    plugin_loader::PluginLoader::CloseDynamicLib(iter->first, iter->second);
    iter++;
  }
  plugin_maps_.clear();
  load_init_ = false;
}

void DeviceContextManager::ClearDeviceContexts() {
  for (auto &iter : device_contexts_) {
    MS_LOG(INFO) << "Release device " << iter.first;
    MS_EXCEPTION_IF_NULL(iter.second);
    iter.second->Destroy();
  }
  device_contexts_.clear();
}

DeviceContext *DeviceContextManager::GetOrCreateDeviceContext(const DeviceContextKey &device_context_key) {
  std::string device_context_key_str = device_context_key.ToString();
  std::string name = device_context_key.device_name_;

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->backend_policy() == "ge") {
    name = "GE";
    device_context_key_str = "GE_0";
  }
  auto device_context_iter = device_contexts_.find(device_context_key_str);
  if (device_context_iter != device_contexts_.end()) {
    return device_context_iter->second.get();
  }

  std::shared_ptr<DeviceContext> device_context;
  auto creator_iter = device_context_creators_.find(name);
  if (creator_iter != device_context_creators_.end()) {
    device_context = (creator_iter->second)(device_context_key);
    MS_EXCEPTION_IF_NULL(device_context);
    device_contexts_[device_context_key_str] = device_context;
  } else {
    MS_LOG(EXCEPTION) << "Create device context failed, please make sure target device:" << name << " is available.";
  }
  return device_context.get();
}

void DeviceContextManager::UpdateDeviceContextKey(const DeviceContextKey &old_key, const DeviceContextKey &new_key) {
  std::string old_key_str = old_key.ToString();
  std::string new_key_str = new_key.ToString();

  auto handle = device_contexts_.extract(old_key_str);
  if (handle.empty()) {
    MS_LOG(EXCEPTION) << "Can not find device context for: " << old_key_str;
  }

  handle.key() = new_key_str;
  (void)device_contexts_.insert(std::move(handle));
}

void DeviceContextManager::WaitTaskFinishOnDevice() const {
  for (const auto &item : device_contexts_) {
    auto device_context = item.second;
    try {
      if (device_context != nullptr && !device_context->device_res_manager_->SyncAllStreams()) {
        MS_LOG(ERROR) << "SyncStream failed";
        return;
      }
    } catch (const std::exception &ex) {
      MS_LOG(ERROR) << "SyncStream failed, exception:" << ex.what();
      return;
    }
  }
}
}  // namespace device
}  // namespace mindspore
