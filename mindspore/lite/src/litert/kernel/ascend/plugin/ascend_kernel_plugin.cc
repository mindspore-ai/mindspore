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

#include <string>
#include "litert/kernel/ascend/plugin/ascend_kernel_plugin.h"
#include "extendrt/cxx_api/dlutils.h"
#include "include/errorcode.h"

namespace mindspore {
AscendKernelPlugin &AscendKernelPlugin::GetInstance() {
  static AscendKernelPlugin instance;
  return instance;
}

AscendKernelPlugin::AscendKernelPlugin() : is_registered_(false) {}

AscendKernelPlugin::~AscendKernelPlugin() { is_registered_ = false; }

int AscendKernelPlugin::Register() {
#if !defined(_WIN32)
  std::lock_guard<std::mutex> locker(mutex_);
  if (is_registered_) {
    return lite::RET_OK;
  }
  Dl_info dl_info;
  dladdr(reinterpret_cast<void *>(this), &dl_info);
  std::string cur_so_path = dl_info.dli_fname;
  auto converter_pos = cur_so_path.find("libmindspore_converter.so");
  if (converter_pos != std::string::npos) {
    MS_LOG(INFO) << "libmindspore_converter.so does not need to register";
    return lite::RET_OK;
  }
  auto pos = cur_so_path.find("libmindspore-lite.so");
  if (pos == std::string::npos) {
    MS_LOG(DEBUG) << "Could not find libmindspore-lite so, cur so path: " << cur_so_path;
    auto c_lite_pos = cur_so_path.find("_c_lite");
    if (c_lite_pos == std::string::npos) {
      MS_LOG(ERROR) << "Could not find _c_lite so, cur so path: " << cur_so_path;
      return lite::RET_ERROR;
    }
    pos = c_lite_pos;
  }
  std::string parent_dir = cur_so_path.substr(0, pos);
  std::string ascend_kernel_plugin_path;
  auto ret = FindSoPath(parent_dir, "libascend_kernel_plugin.so", &ascend_kernel_plugin_path);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Get real path of libascend_kernel_plugin.so failed.";
    return lite::RET_ERROR;
  }
  MS_LOG(INFO) << "Find ascend kernel plugin so success, path = " << ascend_kernel_plugin_path;
  dl_loader_ = std::make_shared<lite::DynamicLibraryLoader>();
  if (dl_loader_ == nullptr) {
    MS_LOG(ERROR) << "Init dynamic library loader failed";
    return lite::RET_ERROR;
  }
  auto status = dl_loader_->Open(ascend_kernel_plugin_path);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "Open libascend_kernel_plugin.so failed";
    return lite::RET_ERROR;
  }
  is_registered_ = true;
#endif
  return lite::RET_OK;
}
}  // namespace mindspore
