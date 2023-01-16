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
#include <set>
#include <fstream>
#include "utils/ms_context.h"
#include "utils/dlopen_macro.h"
#include "utils/os.h"

namespace mindspore {
namespace plugin_loader {
bool PluginLoader::LoadDynamicLib(const std::string &plugin_file, std::map<std::string, void *> *all_handles,
                                  std::stringstream *err_msg) {
  MS_EXCEPTION_IF_NULL(all_handles);
  MS_EXCEPTION_IF_NULL(err_msg);
  void *handle = nullptr;
  std::string err_msg_str;
  auto so_name = GetDynamicLibName(plugin_file);
#if defined(_WIN32) || defined(_WIN64)
  handle = LoadLibraryEx(plugin_file.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
  err_msg_str = std::to_string(GetLastError());
#else
  handle = dlopen(plugin_file.c_str(), RTLD_LAZY | RTLD_LOCAL);
  err_msg_str = GetDlErrorMsg();
#endif
  if (handle == nullptr) {
    MS_LOG(INFO) << "Load dynamic library: " << so_name << " failed. " << err_msg_str;
    *err_msg << "Load dynamic library: " << so_name << " failed. " << err_msg_str << std::endl;
    return false;
  }
  (*all_handles)[so_name] = handle;
  return true;
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
  return target_so;
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
  auto plugin_so_path = cur_so_path.substr(0, pos) + "\\bin";
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
  load_init_ = true;
  MS_EXCEPTION_IF_NULL(MsContext::GetInstance());
  MsContext::GetInstance()->ResisterLoadPluginErrorFunc(
    []() -> std::string { return DeviceContextManager::GetInstance().GetErrorMsg(); });
  if (plugin_path_.empty() && !plugin_loader::PluginLoader::GetPluginPath(&plugin_path_)) {
    MS_LOG(INFO) << "Plugin path is invalid, skip!";
    load_init_ = true;
    dlopen_error_msg_ << "Plugin path is invalid, skip!" << std::endl;
    return;
  }
#ifdef _WIN32
  auto plugin_file = plugin_path_ + "\\mindspore_gpu.dll";
  if (access(plugin_file.c_str(), F_OK) != -1) {
    (void)plugin_loader::PluginLoader::LoadDynamicLib(plugin_file, &plugin_maps_, &dlopen_error_msg_);
  }
#else
  DIR *dir = opendir(plugin_path_.c_str());
  if (dir == nullptr) {
    MS_LOG(ERROR) << "Open plugin dir failed, plugin path:" << plugin_path_;
    load_init_ = true;
    dlopen_error_msg_ << "Open plugin dir failed, plugin path:" << plugin_path_ << std::endl;
    return;
  }
  struct dirent *entry;
  std::map<std::string, std::set<std::string> > multi_version_plugin_map;  // key: plugin name, value: so file name
  while ((entry = readdir(dir)) != nullptr) {
    auto plugin_file = plugin_path_ + PATH_SEPARATOR + entry->d_name;
#ifndef _WIN32
    if (plugin_file.find("libmindspore_") == std::string::npos) {
      continue;
    }
#else
    if (plugin_file.find("mindspore_") == std::string::npos) {
      continue;
    }
#endif
    std::string file_name = entry->d_name;
    auto dot = file_name.find_first_of(".");
    if (dot == std::string::npos) {
      continue;
    }
    multi_version_plugin_map[file_name.substr(0, dot)].insert(plugin_file);
  }
  for (const auto &[plugin_name, file_names] : multi_version_plugin_map) {
    for (auto iter = file_names.rbegin(); iter != file_names.rend();) {
      const auto &file_name = *(iter++);
      auto ret = plugin_loader::PluginLoader::LoadDynamicLib(file_name, &plugin_maps_, &dlopen_error_msg_);
      if (ret) {
        if (iter != file_names.rend()) {
          MS_LOG(INFO) << "Load " << plugin_name << " plugin file " << file_name
                       << " success, skip loading other version.";
        }
        break;
      }
    }
  }
  (void)closedir(dir);
#endif
}

void DeviceContextManager::UnloadPlugin() {
  if (plugin_maps_.empty()) {
    return;
  }
  device_context_creators_.clear();
  auto iter = plugin_maps_.begin();
  while (iter != plugin_maps_.end()) {
    plugin_loader::PluginLoader::CloseDynamicLib(iter->first, iter->second);
    (void)iter++;
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

  if (ms_context->IsDefaultDeviceTarget()) {
    MS_LOG(INFO) << "No context.device_target set, use " << name << " as default.";
  }
  std::shared_ptr<DeviceContext> device_context;
  auto creator_iter = device_context_creators_.find(name);
  if (creator_iter != device_context_creators_.end()) {
    device_context = (creator_iter->second)(device_context_key);
    MS_EXCEPTION_IF_NULL(device_context);
    device_contexts_[device_context_key_str] = device_context;
  } else {
    MS_LOG(EXCEPTION) << "Create device context failed, please make sure target device:" << name
                      << " is available, error message of loading plugins: " << std::endl
                      << GetErrorMsg();
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

std::string DeviceContextManager::GetErrorMsg() const { return dlopen_error_msg_.str(); }
}  // namespace device
}  // namespace mindspore
