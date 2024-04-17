/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "transform/symbol/symbol_utils.h"
#include "transform/symbol/acl_rt_allocator_symbol.h"

namespace mindspore {
namespace transform {
aclrtAllocatorCreateDescFunObj aclrtAllocatorCreateDesc_ = nullptr;
aclrtAllocatorDestroyDescFunObj aclrtAllocatorDestroyDesc_ = nullptr;
aclrtAllocatorRegisterFunObj aclrtAllocatorRegister_ = nullptr;
aclrtAllocatorSetAllocAdviseFuncToDescFunObj aclrtAllocatorSetAllocAdviseFuncToDesc_ = nullptr;
aclrtAllocatorSetAllocFuncToDescFunObj aclrtAllocatorSetAllocFuncToDesc_ = nullptr;
aclrtAllocatorSetFreeFuncToDescFunObj aclrtAllocatorSetFreeFuncToDesc_ = nullptr;
aclrtAllocatorSetGetAddrFromBlockFuncToDescFunObj aclrtAllocatorSetGetAddrFromBlockFuncToDesc_ = nullptr;
aclrtAllocatorSetObjToDescFunObj aclrtAllocatorSetObjToDesc_ = nullptr;
aclrtAllocatorUnregisterFunObj aclrtAllocatorUnregister_ = nullptr;

void LoadAclAllocatorApiSymbol(const std::string &ascend_path) {
  std::string allocator_plugin_path = ascend_path + "lib64/libascendcl.so";
  auto handler = GetLibHandler(allocator_plugin_path);
  if (handler == nullptr) {
    MS_LOG(WARNING) << "Dlopen " << allocator_plugin_path << " failed!" << dlerror();
    return;
  }
  aclrtAllocatorCreateDesc_ = DlsymAscendFuncObj(aclrtAllocatorCreateDesc, handler);
  aclrtAllocatorDestroyDesc_ = DlsymAscendFuncObj(aclrtAllocatorDestroyDesc, handler);
  aclrtAllocatorRegister_ = DlsymAscendFuncObj(aclrtAllocatorRegister, handler);
  aclrtAllocatorSetAllocAdviseFuncToDesc_ = DlsymAscendFuncObj(aclrtAllocatorSetAllocAdviseFuncToDesc, handler);
  aclrtAllocatorSetAllocFuncToDesc_ = DlsymAscendFuncObj(aclrtAllocatorSetAllocFuncToDesc, handler);
  aclrtAllocatorSetFreeFuncToDesc_ = DlsymAscendFuncObj(aclrtAllocatorSetFreeFuncToDesc, handler);
  aclrtAllocatorSetGetAddrFromBlockFuncToDesc_ =
    DlsymAscendFuncObj(aclrtAllocatorSetGetAddrFromBlockFuncToDesc, handler);
  aclrtAllocatorSetObjToDesc_ = DlsymAscendFuncObj(aclrtAllocatorSetObjToDesc, handler);
  aclrtAllocatorUnregister_ = DlsymAscendFuncObj(aclrtAllocatorUnregister, handler);
  MS_LOG(INFO) << "Load acl allocator api success!";
}

}  // namespace transform
}  // namespace mindspore
