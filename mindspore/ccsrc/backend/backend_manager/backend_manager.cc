/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "include/backend/backend_manager/backend_manager.h"
#include <vector>
#include <utility>
#include <map>
#include <memory>
#include <string>
#ifndef _WIN32
#include <libgen.h>
#endif
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/dlopen_macro.h"

namespace mindspore {
namespace backend {
namespace {
size_t custom_backend_num = 0;
constexpr size_t kCustomBackendBeginId = 2;
std::set<std::string> custom_backend_names;

std::map<BackendType, std::string> backend_type_to_lib_name = {{kGEBackend, kGEBackendLibName}};
std::map<BackendName, BackendType> backend_name_to_type = {{kMSBackendName, kMSBackend}, {kGEBackendName, kGEBackend}};
std::map<BackendType, BackendName> backend_type_to_name = {{kMSBackend, kMSBackendName}, {kGEBackend, kGEBackendName}};

BackendName GetBackendNameByType(BackendType backend_type) {
  auto iter = backend_type_to_name.find(backend_type);
  if (iter == backend_type_to_name.end()) {
    MS_LOG(EXCEPTION) << "Invalid backend type: " << backend_type;
  }
  return iter->second;
}

BackendType GetBackendTypeByName(const BackendName &backend_name) {
  BackendType backend_type;
  auto iter = backend_name_to_type.find(backend_name);
  if (iter == backend_name_to_type.end()) {
    if (custom_backend_names.find(backend_name) == custom_backend_names.end()) {
      MS_LOG(EXCEPTION) << "Invalid backend name: " << backend_name << ", you should register custom backend first";
    }
    auto custom_backend_type = static_cast<BackendType>(kCustomBackendBeginId + custom_backend_num);
    if (custom_backend_type >= kInvalidBackend) {
      MS_LOG(EXCEPTION) << "Max backend type is 11, but now custom_backend_type is: " << custom_backend_type << ".";
    }
    ++custom_backend_num;
    backend_name_to_type.insert({backend_name, custom_backend_type});
    backend_type_to_name.insert({custom_backend_type, backend_name});
    backend_type = custom_backend_type;
  } else {
    backend_type = iter->second;
  }
  return backend_type;
}

std::string GetBackendLibNameByType(BackendType backend_type) {
  auto iter = backend_type_to_lib_name.find(backend_type);
  if (iter == backend_type_to_lib_name.end()) {
    MS_LOG(EXCEPTION) << "Invalid backend type for the dynamic load: " << backend_type;
  }
  return iter->second;
}

BackendType GetBackendType(const std::string &backend_name) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto device_target = context->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  // 1. Use custom backend
  if (!backend_name.empty() && backend_name != kMSBackendName && backend_name != kGEBackendName) {
    return GetBackendTypeByName(backend_name);
  }

  // 2. Return kMSBackend when device target is not Ascend
  if (device_target != kAscendDevice) {
    // GE backend is only used for ascend
    return kMSBackend;
  }

  // 3. Use ms_backend or GE when backend_name is specified
  if (!backend_name.empty()) {
    return GetBackendTypeByName(backend_name);
  }

  // 4. Get backend type when backend_name is not specified
  if (context->IsKByKExecutorMode()) {
    return kMSBackend;
  } else {
    return kGEBackend;
  }
}

std::string GetCurrentDir() {
#ifndef _WIN32
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void *>(GetCurrentDir), &dl_info) == 0) {
    MS_LOG(WARNING) << "Get dladdr error";
    return "";
  }
  std::string curr_so_path = dl_info.dli_fname;
  return dirname(curr_so_path.data());
#else
  return "";
#endif
}
}  // namespace

BackendManager &BackendManager::GetInstance() {
  static BackendManager instance{};
  return instance;
}

void BackendManager::Register(const BackendName &backend_name, BackendCreator &&backend_creator) {
#ifndef WITH_BACKEND
  // UT don't load a real custom backend, so add it to the custom_backend_names here.
  if (custom_backend_names.find(backend_name) == custom_backend_names.end()) {
    custom_backend_names.insert(backend_name);
  }
#endif
  auto backend_type = GetBackendType(backend_name);
  if (backend_creators_.find(backend_type) == backend_creators_.end()) {
    (void)backend_creators_.emplace(backend_type, std::move(backend_creator));
  } else {
    MS_LOG(EXCEPTION) << "Backend name: " << backend_name << " has been registered.";
  }
}

void BackendManager::Clear() {
  for (size_t i = 0; i < kInvalidBackend; i++) {
    if (backends_[i] != nullptr) {
      backends_[i]->Clear();
      backends_[i] = nullptr;
    }
  }

  backend_creators_.clear();
  backend_load_handle_.clear();
}

void BackendManager::ClearGraph(BackendType backend_type, BackendGraphId backend_graph_id) {
  if (backend_type >= kInvalidBackend) {
    MS_LOG(EXCEPTION) << "Invalid backend type: " << backend_type;
  }
  auto backend = backends_[static_cast<int>(backend_type)];
  // when process exits, backend is cleared in ClearResAtexit()
  if (backend == nullptr) {
    return;
  }
  backend->ClearGraph(backend_graph_id);
}

std::vector<GraphFragmentPtr> BackendManager::Split(const FuncGraphPtr &func_graph, const std::string &backend_name) {
  auto backend_type = GetBackendType(backend_name);
  auto backend = GetOrCreateBackend(backend_type);
  MS_EXCEPTION_IF_NULL(backend);
  return backend->Split(func_graph);
}

std::pair<BackendType, BackendGraphId> BackendManager::Build(const FuncGraphPtr &func_graph,
                                                             const BackendJitConfig &backend_jit_config,
                                                             const std::string &backend_name) {
  MS_EXCEPTION_IF_NULL(func_graph);
  // Convert the dry run to kernel luanch skip for runtime.
  if (common::IsCompileSimulation()) {
    common::SetEnv("MS_KERNEL_LAUNCH_SKIP", "all", 1);
  }
  auto backend_type = GetBackendType(backend_name);
  auto backend = GetOrCreateBackend(backend_type);
  MS_EXCEPTION_IF_NULL(backend);
  auto graph_id = backend->Build(func_graph, backend_jit_config);
  MS_LOG(INFO) << "Backend build graph, backend name: " << backend_name << ", backend type: " << backend_type
               << ", backend graph id: " << graph_id;
  return {backend_type, graph_id};
}

RunningStatus BackendManager::Run(BackendType backend_type, BackendGraphId graph_id, const VectorRef &inputs,
                                  VectorRef *outputs) {
  auto backend = backends_[backend_type];
  MS_EXCEPTION_IF_NULL(backend);
  MS_LOG(INFO) << "Backend run graph: " << graph_id << ", backend type: " << backend_type;
  return backend->Run(graph_id, inputs, outputs);
}

string BackendManager::ExportIR(const FuncGraphPtr &anf_graph, const std::string &file_name, bool is_save_to_file,
                                IRFormat ir_format, const std::string &backend_name) {
  auto backend_type = GetBackendType(backend_name);
  auto backend = GetOrCreateBackend(backend_type);
  MS_EXCEPTION_IF_NULL(backend);
  return backend->ExportIR(anf_graph, file_name, is_save_to_file, ir_format);
}

void BackendManager::ConvertIR(const FuncGraphPtr &anf_graph,
                               const std::map<std::string, std::shared_ptr<tensor::Tensor>> &init_tensors,
                               IRFormat ir_format, const std::string &backend_name) {
  auto backend_type = GetBackendType(backend_name);
  auto backend = GetOrCreateBackend(backend_type);
  MS_EXCEPTION_IF_NULL(backend);
  return backend->ConvertIR(anf_graph, init_tensors, ir_format);
}

bool BackendManager::LoadBackend(const BackendName &backend_name, const std::string &backend_path) {
  if (backend_name == kMSBackendName) {
    MS_LOG(EXCEPTION) << "MS backend is bulit-in backend, don't support the dynamic load.";
  }
  if (custom_backend_names.find(backend_name) == custom_backend_names.end()) {
    custom_backend_names.insert(backend_name);
  }
  auto backend_type = GetBackendType(backend_name);
  if (backend_load_handle_.count(backend_type) > 0) {
    return true;
  }

  MS_LOG(INFO) << "Backendmanager dlopen backend lib name: " << backend_name;
  void *handle;
  std::string err_msg = "";
  std::string cur_backend_lib_name;
  std::lock_guard<std::mutex> lock(backend_mutex_);
#ifndef _WIN32
  if (backend_name != kGEBackendName) {
    cur_backend_lib_name = backend_path;
  } else {
    cur_backend_lib_name = GetCurrentDir() + "/" + kGEBackendLibName;
  }
  MS_LOG(INFO) << "Backendmanager dlopen current backend lib name: " << cur_backend_lib_name;
  handle = dlopen(cur_backend_lib_name.c_str(), RTLD_LAZY);
  err_msg = GetDlErrorMsg();
#else
  handle = LoadLibrary(cur_backend_lib_name.c_str());
  err_msg = std::to_string(GetLastError());
#endif
  if (cur_backend_lib_name.empty()) {
    MS_LOG(ERROR) << "Backend path: " << cur_backend_lib_name << " is empty.";
    return false;
  }
  if (handle == nullptr) {
    MS_LOG(ERROR) << "Loading " + cur_backend_lib_name + " failed. Error: " + err_msg;
    return false;
  }
  (void)backend_load_handle_.emplace(backend_type, handle);
  backend_type_to_lib_name.insert({backend_type, cur_backend_lib_name});
  return true;
}

void BackendManager::UnloadBackend() {
  for (auto iter : backend_load_handle_) {
    auto backend_lib_name = GetBackendLibNameByType(iter.first);
    auto handle = iter.second;
#ifndef _WIN32
    if (dlclose(handle) != 0) {
      MS_LOG(EXCEPTION) << "Closing " + backend_lib_name + " handle failed. Error: " + GetDlErrorMsg();
    }
#else
    if (!FreeLibrary(reinterpret_cast<HINSTANCE__ *>(handle))) {
      MS_LOG(EXCEPTION) << "Closing " + backend_lib_name + " handle failed. Error: " + std::to_string(GetLastError());
    }
#endif
  }
}

BackendBase *BackendManager::GetOrCreateBackend(const BackendType &backend_type) {
  if (backends_[backend_type] != nullptr) {
    return backends_[backend_type].get();
  }

  // Only the ge backend and custom backend support the dynamic load, custom backend has been loaded before.
  if (backend_type == kGEBackend) {
    auto ret = LoadBackend(kGEBackendName);
    if (!ret) {
      MS_LOG(EXCEPTION) << "Load backend failed, please make sure the backend:" << backend_type << " is correct.";
    }
  }

  auto creator_iter = backend_creators_.find(backend_type);
  if (creator_iter == backend_creators_.end()) {
    MS_LOG(EXCEPTION)
      << "Create backend failed, please make sure the backend:" << GetBackendNameByType(backend_type)
      << " has been registered. If you want to use the custom backend, please register it first and keep the "
         "name same as the register_custom_backend api.";
  }

  MS_LOG(INFO) << "The created backend type: " << backend_type;
  auto backend = (creator_iter->second)();
  MS_EXCEPTION_IF_NULL(backend);
  backends_[backend_type] = backend;
  return backend.get();
}
}  // namespace backend
}  // namespace mindspore
