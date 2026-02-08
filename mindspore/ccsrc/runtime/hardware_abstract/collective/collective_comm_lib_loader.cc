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

#include "include/runtime/hardware_abstract/collective/collective_comm_lib_loader.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace device {
bool CollectiveCommLibLoader::Initialize() {
  std::string err_msg = "";
#ifndef _WIN32
  collective_comm_lib_ptr_ = dlopen(comm_lib_name_.c_str(), RTLD_LAZY);
  err_msg = GetDlErrorMsg();
#else
  collective_comm_lib_ptr_ = LoadLibrary(comm_lib_name_.c_str());
  err_msg = std::to_string(GetLastError());
#endif
  if (collective_comm_lib_ptr_ == nullptr) {
    MS_LOG(EXCEPTION) << "Loading " + comm_lib_name_ + " failed. Error: " + err_msg;
  }
  return true;
}

bool CollectiveCommLibLoader::Finalize() {
  MS_EXCEPTION_IF_NULL(collective_comm_lib_ptr_);

#ifndef _WIN32
  if (dlclose(collective_comm_lib_ptr_) != 0) {
    MS_LOG(EXCEPTION) << "Closing " + comm_lib_name_ + " handle failed. Error: " + GetDlErrorMsg();
  }
#else
  if (!FreeLibrary(reinterpret_cast<HINSTANCE__ *>(collective_comm_lib_ptr_))) {
    MS_LOG(EXCEPTION) << "Closing " + comm_lib_name_ + " handle failed. Error: " + std::to_string(GetLastError());
  }
#endif
  return true;
}
}  // namespace device
}  // namespace mindspore
