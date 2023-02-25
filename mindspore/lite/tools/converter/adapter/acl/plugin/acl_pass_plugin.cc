/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/adapter/acl/plugin/acl_pass_plugin.h"
#include "utils/ms_utils.h"
#include "utils/log_adapter.h"
#include "include/errorcode.h"
#if !defined(_WIN32)
#include <dlfcn.h>
#include "extendrt/cxx_api/dlutils.h"
#endif

namespace mindspore {
namespace opt {
std::mutex AclPassPlugin::mutex_;

AclPassPlugin::AclPassPlugin() = default;

AclPassPlugin::~AclPassPlugin() {
#if !defined(_WIN32)
  MS_LOG(DEBUG) << "~AclPassPlugin() begin.";
  if (handle_ != nullptr) {
    (void)dlclose(handle_);
    handle_ = nullptr;
    creator_func_ = nullptr;
  }
  MS_LOG(DEBUG) << "~AclPassPlugin() end.";
#endif
}

bool AclPassPlugin::GetPluginSoPath() {
#if !defined(_WIN32)
  Dl_info dl_info;
  dladdr(reinterpret_cast<void *>(this), &dl_info);
  std::string cur_so_path = dl_info.dli_fname;
  auto pos = cur_so_path.find("libmindspore_converter.so");
  if (pos == std::string::npos) {
    MS_LOG(ERROR) << "Could not find libmindspore_converter so, cur so path: " << cur_so_path;
    return false;
  }
  std::string parent_dir = cur_so_path.substr(0, pos);
  std::string ascend_pass_plugin_path;
  auto ret = FindSoPath(parent_dir, "libascend_pass_plugin.so", &ascend_pass_plugin_path);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Get real path of libascend_pass_plugin.so failed.";
    return false;
  }
  if (ret == kSuccess && !ascend_pass_plugin_path.empty()) {
    real_path_ = ascend_pass_plugin_path;
    MS_LOG(INFO) << "Find ascend pass plugin so success, path = " << real_path_;
    return true;
  }
#endif
  return false;
}

std::shared_ptr<Pass> AclPassPlugin::CreateAclPass(const std::shared_ptr<ConverterPara> &param) {
  std::lock_guard<std::mutex> lock(mutex_);
  static AclPassPlugin instance;
  return instance.CreateAclPassInner(param);
}

std::shared_ptr<Pass> AclPassPlugin::CreateAclPassInner(const std::shared_ptr<ConverterPara> &param) {
#if !defined(_WIN32)
  if (creator_func_ == nullptr) {
    if (!GetPluginSoPath() || real_path_.empty()) {
      MS_LOG(ERROR) << "Failed to get path of libascend_pass_plugin.so";
      return nullptr;
    }
    void *function = nullptr;
    auto ret = DLSoOpen(real_path_, "CreateAclPass", &handle_, &function);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "DLSoOpen failed, so path: " << real_path_ << ", ret: " << ret;
      return nullptr;
    }
    creator_func_ = reinterpret_cast<AclPassCreatorFunc>(function);
    if (creator_func_ == nullptr) {
      MS_LOG(ERROR) << "Cast symbol CreateAclPass failed.";
      return nullptr;
    }
  }
  auto pass_ptr = creator_func_(param);
  if (pass_ptr == nullptr) {
    MS_LOG(ERROR) << "Failed to call CreateAclPass.";
    return nullptr;
  }
  return std::shared_ptr<Pass>(pass_ptr);
#else
  MS_LOG(ERROR) << "Not support load libascend_pass_plugin.so in Windows";
  return nullptr;
#endif
}
}  // namespace opt
}  // namespace mindspore
