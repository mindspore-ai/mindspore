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

AscendKernelPlugin::~AscendKernelPlugin() {}

int AscendKernelPlugin::Register() {
  if (is_registered_) {
    return lite::RET_OK;
  }
  std::string dl_so_path;
  auto get_path_ret = DLSoPath({"libmindspore-lite.so"}, "libascend_kernel_plugin.so", &dl_so_path);
  if (get_path_ret != kSuccess) {
    MS_LOG(ERROR) << "Get libascend_kernel_plugin.so path failed";
    return lite::RET_ERROR;
  }
  dl_loader_ = std::make_shared<lite::DynamicLibraryLoader>();
  if (dl_loader_ == nullptr) {
    MS_LOG(ERROR) << "Init dynamic library loader failed";
    return lite::RET_ERROR;
  }
  auto status = dl_loader_->Open(dl_so_path);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "Open libascend_kernel_plugin.so failed";
    return lite::RET_ERROR;
  }
  is_registered_ = true;
  return lite::RET_OK;
}
}  // namespace mindspore
