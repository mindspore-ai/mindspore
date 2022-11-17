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
AclPassPlugin &AclPassPlugin::GetInstance() {
  static AclPassPlugin instance;
  return instance;
}

AclPassPlugin::AclPassPlugin() : handle_(nullptr), pass_ptr_(nullptr) {}

bool AclPassPlugin::HasPluginSo() {
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

Pass *AclPassPlugin::CreateAclPass(const std::shared_ptr<ConverterPara> &param) {
#if !defined(_WIN32)
  if (pass_ptr_ != nullptr) {
    MS_LOG(INFO) << "Acl pass has been created.";
    return pass_ptr_;
  }
  void *function = nullptr;
  auto ret = DLSoOpen(real_path_, "CreateAclPass", &handle_, &function);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "DLSoOpen failed, so path: " << real_path_ << ", ret: " << ret;
    return nullptr;
  }
  auto create_func = reinterpret_cast<mindspore::opt::Pass *(*)(const std::shared_ptr<ConverterPara> &)>(function);
  if (create_func == nullptr) {
    MS_LOG(ERROR) << "Cast symbol CreateAclPass failed.";
    return nullptr;
  }
  pass_ptr_ = create_func(param);
  if (pass_ptr_ == nullptr) {
    MS_LOG(ERROR) << "New acl pass failed.";
    return nullptr;
  }
#endif
  return pass_ptr_;
}

void AclPassPlugin::DestroyAclPass(Pass *acl_pass) {
#if !defined(_WIN32)
  if (handle_ == nullptr) {
    MS_LOG(ERROR) << "Handle is nullptr .";
    return;
  }
  if (acl_pass != pass_ptr_) {
    MS_LOG(ERROR) << "Out pass ptr is not same as inner pass ptr.";
    return;
  }
  auto destroy_func = reinterpret_cast<void (*)(mindspore::opt::Pass *)>(dlsym(handle_, "DestroyAclPass"));
  if (destroy_func == nullptr) {
    MS_LOG(ERROR) << "Undefined symbol DestroyAclPass in ['libascend_pass_plugin.so']";
    return;
  }
  destroy_func(acl_pass);
  pass_ptr_ = nullptr;
#endif
}

AclPassPlugin::~AclPassPlugin() {
#if !defined(_WIN32)
  MS_LOG(DEBUG) << "~AclPassPlugin() begin.";
  if (handle_ != nullptr) {
    (void)dlclose(handle_);
    handle_ = nullptr;
  }
  MS_LOG(DEBUG) << "~AclPassPlugin() end.";
#endif
}
}  // namespace opt
}  // namespace mindspore
