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
#ifdef __linux__
#include <sys/wait.h>
#endif  // #ifdef __linux__
#include <dirent.h>
#include <algorithm>
#include <string>
#include <set>
#include <fstream>
#include "utils/ms_context.h"
#include "utils/dlopen_macro.h"
#include "utils/os.h"

namespace mindspore {
namespace {
size_t constexpr GetStrLen(const char *const str) {
  if (*str == '\0') {
    return 0;
  } else {
    return GetStrLen(str + 1) + 1;
  }
}

constexpr auto kCudaHomeEnv = "CUDA_HOME";
constexpr auto kNvccVersionKeyWords = "Cuda compilation tools, release ";
constexpr size_t kNvccVersionKeyWordsSize = GetStrLen(kNvccVersionKeyWords);
constexpr auto kSuccessKeyWord = "Success";
constexpr size_t kSuccessKeyWordSize = GetStrLen(kSuccessKeyWord);
constexpr size_t kBufferSize = 999;
constexpr auto kGpuPluginName = "libmindspore_gpu";

#ifdef __linux__
class FdScope {
 public:
  explicit FdScope(int fd) : fd_(fd) {}
  ~FdScope() { (void)close(fd_); }

 private:
  int fd_;
};

std::string GetNvccRealPath(const std::string &cuda_path) {
  auto nvcc_path = cuda_path + "/bin/nvcc";
  char real_path_buffer[PATH_MAX];
  if (realpath(nvcc_path.c_str(), real_path_buffer) == nullptr) {
    MS_LOG(WARNING) << "Invalid environment variable CUDA_HOME [" << cuda_path << "], can not find nvcc file ["
                    << nvcc_path << "], please check the CUDA_HOME.";
    return "";
  }
  return real_path_buffer;
}

std::string GetCudaVersionFromNvcc(const std::string &nvcc_path) {
  int pipe_fd[2];
  if (pipe(pipe_fd) != 0) {
    MS_LOG(ERROR) << "Create pipe failed, ret = " << errno << ", reason = " << strerror(errno);
    return "";
  }
  FdScope fd0(pipe_fd[0]);
  FdScope fd1(pipe_fd[1]);
  pid_t pid = fork();
  if (pid < 0) {
    MS_LOG(ERROR) << "Fork child process failed, ret = " << errno << ", reason = " << strerror(errno);
    return "";
  } else if (pid == 0) {  // child process
    (void)dup2(pipe_fd[1], STDOUT_FILENO);
    MS_LOG(DEBUG) << "Start exec " << nvcc_path << " --version";
    if (execl(nvcc_path.c_str(), "nvcc", "--version", nullptr) == -1) {
      MS_LOG(ERROR) << "Get cuda version from " << nvcc_path << " failed, ret = " << errno
                    << ", reason = " << strerror(errno);
      exit(-1);
    }
  } else {  // parent process
    MS_LOG(DEBUG) << "Child process NVCC pid = " << pid;
    int status;
    std::string buffer(kBufferSize, 0);
    if (waitpid(pid, &status, 0) == -1) {
      MS_LOG(ERROR) << "Wait child process failed, ret = " << errno << ", reason = " << strerror(errno);
      return "";
    }
    if (auto read_size = read(pipe_fd[0], buffer.data(), buffer.size()); read_size <= 0) {
      MS_LOG(WARNING) << "Read from pipe failed, ret = " << errno << ", reason = " << strerror(errno);
      return "";
    } else {
      buffer.resize(read_size);
    }

    MS_LOG(DEBUG) << "Child process return: " << buffer;
    auto pos = buffer.find(kNvccVersionKeyWords);
    if (pos == std::string::npos) {
      MS_LOG(ERROR) << "Cannot found nvcc version key words [" << kNvccVersionKeyWords << "], nvcc return: " << buffer;
      return "";
    }
    auto tmp_str = buffer.substr(pos + kNvccVersionKeyWordsSize);
    pos = tmp_str.find_first_of(',');
    if (pos == std::string::npos) {
      MS_LOG(ERROR) << "Cannot found nvcc version key word \',\', nvcc return: " << tmp_str;
      return "";
    }
    auto version_str = tmp_str.substr(0, pos);
    MS_LOG(INFO) << "Get cuda version [" << version_str << "] from env CUDA_HOME.";
    return version_str;
  }
  return "";  // useless code makes static checking tools happy.
}

// only support version str that format is "a.b"
bool GetIntVersionFromVersionStr(const std::string &version_str, size_t *major, size_t *minor) {
  MS_EXCEPTION_IF_NULL(major);
  MS_EXCEPTION_IF_NULL(minor);
  size_t major_num = 0;
  size_t minor_num = 0;
  auto dot_pos = version_str.find('.');
  if (dot_pos == std::string::npos) {
    return false;
  }
  std::string minor_str = version_str.substr(dot_pos + 1);
  std::string major_str = version_str.substr(0, dot_pos);
  try {
    major_num = std::stoull(major_str);
    minor_num = std::stoull(minor_str);
  } catch (...) {
    return false;
  }
  *major = major_num;
  *minor = minor_num;
  return true;
}

bool GetVersionFromFileName(const std::string &file_name, size_t *major, size_t *minor) {
  MS_EXCEPTION_IF_NULL(major);
  MS_EXCEPTION_IF_NULL(minor);
  auto dot_pos = file_name.find_last_of('.');
  if (dot_pos == std::string::npos) {
    return false;
  }
  std::string minor_str = file_name.substr(dot_pos + 1);
  std::string remain_str = file_name.substr(0, dot_pos);
  dot_pos = remain_str.find_last_of('.');
  if (dot_pos == std::string::npos) {
    return false;
  }
  std::string major_str = file_name.substr(dot_pos + 1);
  if (!std::any_of(minor_str.begin(), minor_str.end(), [](char c) { return std::isdigit(c); })) {
    return false;
  }
  if (!std::any_of(major_str.begin(), major_str.end(), [](char c) { return std::isdigit(c); })) {
    return false;
  }
  return GetIntVersionFromVersionStr(major_str + "." + minor_str, major, minor);
}

float VersionToFloat(size_t major, size_t minor) {
  return SizeToFloat(major) + SizeToFloat(minor) / (SizeToFloat(std::to_string(minor).size()) + 1);
}

// dlopen-ing a shared library will find dependency and then relocate for symbols, when relocate failed and a
// "undefined reference to xxxx" occurred, glibc will not rollback the relocation, some relocated symbols will be bound
// to a un-dlopen-ed library when dlopen other libraries in the future, which will cause incomprehensible errors.
// So fork a child process to test whether the library can be loaded, and exit the child process if failed.
bool TestLoadDynamicLib(const std::string &plugin_file, std::string *err_msg) {
  MS_EXCEPTION_IF_NULL(err_msg);
  int pipe_fd[2];
  if (pipe(pipe_fd) != 0) {
    MS_LOG(WARNING) << "Create pipe failed, ret = " << errno << ", reason = " << strerror(errno);
    return false;
  }
  FdScope fd0(pipe_fd[0]);
  FdScope fd1(pipe_fd[1]);
  pid_t pid = fork();
  if (pid < 0) {
    MS_LOG(WARNING) << "Fork child process failed, ret = " << errno << ", reason = " << strerror(errno);
    return false;
  } else if (pid == 0) {  // child process
    // don't care logs of child process, dup stdout/stderr to /dev/null
    int null_fd = open("/dev/null", O_RDWR, 0);
    if (null_fd == -1) {
      MS_LOG(WARNING) << "Child process open /dev/null failed, ret = " << errno << ", reason = " << strerror(errno);
      exit(-1);
    }
    FdScope null(null_fd);
    (void)dup2(null_fd, STDOUT_FILENO);
    (void)dup2(null_fd, STDERR_FILENO);
    // try to dlopen
    void *handle = dlopen(plugin_file.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr) {
      std::string err_msg_str = GetDlErrorMsg();
      auto ret = write(pipe_fd[1], err_msg_str.c_str(), err_msg_str.size());
      (void)ret;  // write(...) has __wur attr, get return value to make compiler happy.
      exit(-1);
    }
    (void)dlclose(handle);
    if (write(pipe_fd[1], kSuccessKeyWord, kSuccessKeyWordSize) <= 0) {
      exit(-1);
    }
    exit(0);
  } else {  // parent process
    MS_LOG(DEBUG) << "Child process dlopen pid = " << pid;
    int status;
    std::string buffer(kBufferSize, 0);
    if (waitpid(pid, &status, 0) == -1) {
      MS_LOG(ERROR) << "Wait child process failed, ret = " << errno << ", reason = " << strerror(errno);
      return false;
    }
    if (auto read_size = read(pipe_fd[0], buffer.data(), buffer.size()); read_size <= 0) {
      MS_LOG(WARNING) << "Read from pipe failed, ret = " << errno << ", reason = " << strerror(errno);
      return false;
    } else {
      buffer.resize(read_size);
    }

    MS_LOG(DEBUG) << "Child process return: " << buffer;
    if (std::string(buffer.c_str()) == kSuccessKeyWord) {
      return true;
    } else {
      *err_msg = buffer;
      return false;
    }
  }
  return false;  // useless code makes static checking tools happy.
}
#endif  // #ifdef __linux__
}  // namespace
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
#elif defined(__linux__)
  if (TestLoadDynamicLib(plugin_file, &err_msg_str)) {
    handle = dlopen(plugin_file.c_str(), RTLD_LAZY | RTLD_LOCAL);
  }
#else  // macos
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
const DeviceContext *FetchRealDeviceContext(const CNodePtr &node, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(device_context);

  if (!common::AnfAlgo::HasNodeAttr(kAttrPrimitiveTarget, node)) {
    return device_context;
  }
  const auto &target = common::AnfAlgo::GetNodeAttr<std::string>(node, kAttrPrimitiveTarget);
  if (target == device_context->device_context_key().device_name_) {
    return device_context;
  }

  const auto &real_device_context = DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {target, device_context->device_context_key().device_id_});
  MS_EXCEPTION_IF_NULL(real_device_context);
  real_device_context->Initialize();
  return real_device_context;
}

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
  std::map<std::string, std::set<std::string>> multi_version_plugin_map;  // key: plugin name, value: so file name
  while ((entry = readdir(dir)) != nullptr) {
    auto plugin_file = plugin_path_ + PATH_SEPARATOR + entry->d_name;
    if (plugin_file.find("libmindspore_") == std::string::npos) {
      continue;
    }
    std::string file_name = entry->d_name;
    auto dot = file_name.find_first_of(".");
    if (dot == std::string::npos) {
      continue;
    }
    multi_version_plugin_map[file_name.substr(0, dot)].insert(plugin_file);
  }

  for (const auto &[plugin_name, file_names] : multi_version_plugin_map) {
    if (plugin_name == kGpuPluginName) {
      std::string cuda_home = common::GetEnv(kCudaHomeEnv);
      if (!cuda_home.empty() && file_names.size() > 1) {
        SelectGpuPlugin(cuda_home, file_names);
        continue;
      }
    }
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
}

void DeviceContextManager::ClearDeviceContexts() {
  for (auto &iter : device_contexts_) {
    MS_LOG(INFO) << "Release device " << iter.first;
    MS_EXCEPTION_IF_NULL(iter.second);
    iter.second->Destroy();
  }
  device_contexts_.clear();
}

void DeviceContextManager::BindDeviceCtx() const {
  for (auto &iter : device_contexts_) {
    MS_EXCEPTION_IF_NULL(iter.second);
    MS_EXCEPTION_IF_NULL(iter.second->device_res_manager_);
    iter.second->device_res_manager_->BindDeviceToCurrentThread(true);
  }
}

DeviceContext *DeviceContextManager::GetOrCreateDeviceContext(const DeviceContextKey &device_context_key,
                                                              string jit_level /* ="" */) {
  std::string device_context_key_str = device_context_key.ToString();
  std::string name = device_context_key.device_name_;

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->backend_policy() == "ge" && (jit_level == kAttrJitLevelO3 || jit_level == "")) {
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

void DeviceContextManager::SelectGpuPlugin(const std::string &cuda_home, const std::set<std::string> &file_names) {
#ifdef __linux__
  auto nvcc_path = GetNvccRealPath(cuda_home);
  if (nvcc_path.empty()) {
    return;
  }
  auto cuda_version = GetCudaVersionFromNvcc(nvcc_path);
  if (cuda_version.empty()) {
    return;
  }
  size_t target_major = 0;
  size_t target_minor = 0;
  if (!GetIntVersionFromVersionStr(cuda_version, &target_major, &target_minor)) {
    MS_LOG(EXCEPTION) << "Get version num from version string " << cuda_version << " failed.";
  }

  std::string selected_plugin = "";
  std::vector<std::pair<size_t, size_t>> all_plugin_version;
  std::vector<std::string> all_plugin_path;
  std::for_each(file_names.begin(), file_names.end(),
                [&selected_plugin, &all_plugin_version, &all_plugin_path, target_major,
                 target_minor](const std::string &file_name) {
                  size_t current_major = 0;
                  size_t current_minor = 0;
                  if (GetVersionFromFileName(file_name, &current_major, &current_minor)) {
                    all_plugin_version.emplace_back(current_major, current_minor);
                    all_plugin_path.emplace_back(file_name);
                  }
                  if (current_major == target_major && current_minor == target_minor) {
                    selected_plugin = file_name;
                  }
                });

  if (selected_plugin.empty()) {
    for (size_t i = 0; i < all_plugin_version.size(); ++i) {
      if (target_major != all_plugin_version[i].first) {
        continue;
      }
      if (VersionToFloat(target_major, target_minor) >
            VersionToFloat(all_plugin_version[i].first, all_plugin_version[i].second) &&
          (i + 1 >= all_plugin_version.size() ||
           VersionToFloat(target_major, target_minor) <
             VersionToFloat(all_plugin_version[i + 1].first, all_plugin_version[i + 1].second))) {
        selected_plugin = all_plugin_path[i];
      }
    }
  }

  if (selected_plugin.empty()) {
    MS_LOG(WARNING) << "Env CUDA_HOME is " << cuda_home << ", but can not find suitable gpu plugin.";
    return;
  }

  auto ret = plugin_loader::PluginLoader::LoadDynamicLib(selected_plugin, &plugin_maps_, &dlopen_error_msg_);
  if (!ret) {
    MS_LOG(WARNING) << "Env CUDA_HOME is " << cuda_home
                    << ", but dlopen file_name failed, reason: " << dlopen_error_msg_.str();
    return;
  }
#endif  // #ifdef __linux__
}
}  // namespace device
}  // namespace mindspore
