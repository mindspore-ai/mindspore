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

#include "minddata/dataset/plugin/plugin_loader.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <vector>

#include "minddata/dataset/plugin/shared_lib_util.h"

namespace mindspore {
namespace dataset {
PluginLoader *PluginLoader::GetInstance() noexcept {
  static PluginLoader pl;
  return &pl;
}

PluginLoader::~PluginLoader() {
  std::vector<std::string> keys;
  // get the keys from map, this is to avoid concurrent iteration and delete
  std::transform(plugins_.begin(), plugins_.end(), std::back_inserter(keys), [](const auto &p) { return p.first; });
  for (std::string &key : keys) {
    Status rc = UnloadPlugin(key);
    MSLOG_IF(MsLogLevel::kError, rc.IsError(), mindspore::NoExceptionType, nullptr) << rc.ToString();
  }
}

// LoadPlugin() is NOT thread-safe. It is supposed to be called when Ops are being built. E.g. PluginOp should call this
// within constructor instead of in its Compute() which is parallel.
Status PluginLoader::LoadPlugin(const std::string &filename, plugin::PluginManagerBase **singleton_plugin) {
  RETURN_UNEXPECTED_IF_NULL(singleton_plugin);
  auto itr = plugins_.find(filename);
  // return ok if this module is already loaded
  if (itr != plugins_.end()) {
    *singleton_plugin = itr->second.first;
    return Status::OK();
  }
  // Open the .so file
  void *handle = SharedLibUtil::Load(filename);
  CHECK_FAIL_RETURN_UNEXPECTED(handle != nullptr,
                               "[Internal ERROR] Fail to load:" + filename + ".\n" + SharedLibUtil::ErrMsg());

  // Load GetInstance function ptr from the so file, so needs to be compiled with -fPIC
  void *func_handle = SharedLibUtil::FindSym(handle, "GetInstance");
  CHECK_FAIL_RETURN_UNEXPECTED(func_handle != nullptr,
                               "[Internal ERROR] Fail to find GetInstance()\n" + SharedLibUtil::ErrMsg());

  // cast the returned function ptr of type void* to the type of GetInstance
  plugin::PluginManagerBase *(*get_instance)(plugin::MindDataManagerBase *) =
    reinterpret_cast<plugin::PluginManagerBase *(*)(plugin::MindDataManagerBase *)>(func_handle);
  RETURN_UNEXPECTED_IF_NULL(get_instance);

  *singleton_plugin = get_instance(nullptr);  // call function ptr to get instance
  RETURN_UNEXPECTED_IF_NULL(*singleton_plugin);

  std::string v1 = (*singleton_plugin)->GetPluginVersion();
  std::string v2(plugin::kSharedIncludeVersion);
  if (v1 != v2) {
    std::string err_msg = "[Internal ERROR] expected:" + v2 + ", received:" + v1 + " please recompile.";
    if (SharedLibUtil::Close(handle) != 0) {
      err_msg += ("\ndlclose() error, err_msg:" + SharedLibUtil::ErrMsg() + ".");
    }
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  const std::map<std::string, std::set<std::string>> module_names = (*singleton_plugin)->GetModuleNames();
  for (auto &p : module_names) {
    std::string msg = "Plugin " + p.first + " has module:";
    MS_LOG(DEBUG) << std::accumulate(p.second.begin(), p.second.end(), msg,
                                     [](const std::string &msg, const std::string &nm) { return msg + " " + nm; });
  }

  // save the name and handle
  std::pair<plugin::PluginManagerBase *, void *> plugin_new = std::make_pair(*singleton_plugin, handle);
  plugins_.insert({filename, plugin_new});
  return Status::OK();
}

Status PluginLoader::UnloadPlugin(const std::string &filename) {
  auto itr = plugins_.find(filename);
  RETURN_OK_IF_TRUE(itr == plugins_.end());  // return true if this plugin was never loaded or already removed

  void *func_handle = SharedLibUtil::FindSym(itr->second.second, "DestroyInstance");
  CHECK_FAIL_RETURN_UNEXPECTED(func_handle != nullptr,
                               "[Internal ERROR] Fail to find DestroyInstance()\n" + SharedLibUtil::ErrMsg());

  void (*destroy_instance)() = reinterpret_cast<void (*)()>(func_handle);
  RETURN_UNEXPECTED_IF_NULL(destroy_instance);

  destroy_instance();
  CHECK_FAIL_RETURN_UNEXPECTED(SharedLibUtil::Close(itr->second.second) == 0,
                               "[Internal ERROR] dlclose() error: " + SharedLibUtil::ErrMsg());

  plugins_.erase(filename);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
