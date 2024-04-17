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
#ifndef MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_RT_ALLOCATOR_SYMBOL_H_
#define MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_RT_ALLOCATOR_SYMBOL_H_
#include <string>
#include "acl/acl_rt_allocator.h"
#include "utils/dlopen_macro.h"

namespace mindspore {
namespace transform {
ORIGIN_METHOD(aclrtAllocatorCreateDesc, aclrtAllocatorDesc)
ORIGIN_METHOD(aclrtAllocatorDestroyDesc, aclError, aclrtAllocatorDesc)
ORIGIN_METHOD(aclrtAllocatorRegister, aclError, aclrtStream, aclrtAllocatorDesc)
ORIGIN_METHOD(aclrtAllocatorSetAllocAdviseFuncToDesc, aclError, aclrtAllocatorDesc, aclrtAllocatorAllocAdviseFunc)
ORIGIN_METHOD(aclrtAllocatorSetAllocFuncToDesc, aclError, aclrtAllocatorDesc, aclrtAllocatorAllocFunc)
ORIGIN_METHOD(aclrtAllocatorSetFreeFuncToDesc, aclError, aclrtAllocatorDesc, aclrtAllocatorFreeFunc)
ORIGIN_METHOD(aclrtAllocatorSetGetAddrFromBlockFuncToDesc, aclError, aclrtAllocatorDesc,
              aclrtAllocatorGetAddrFromBlockFunc)
ORIGIN_METHOD(aclrtAllocatorSetObjToDesc, aclError, aclrtAllocatorDesc, aclrtAllocator)
ORIGIN_METHOD(aclrtAllocatorUnregister, aclError, aclrtStream)

extern aclrtAllocatorCreateDescFunObj aclrtAllocatorCreateDesc_;
extern aclrtAllocatorDestroyDescFunObj aclrtAllocatorDestroyDesc_;
extern aclrtAllocatorRegisterFunObj aclrtAllocatorRegister_;
extern aclrtAllocatorSetAllocAdviseFuncToDescFunObj aclrtAllocatorSetAllocAdviseFuncToDesc_;
extern aclrtAllocatorSetAllocFuncToDescFunObj aclrtAllocatorSetAllocFuncToDesc_;
extern aclrtAllocatorSetFreeFuncToDescFunObj aclrtAllocatorSetFreeFuncToDesc_;
extern aclrtAllocatorSetGetAddrFromBlockFuncToDescFunObj aclrtAllocatorSetGetAddrFromBlockFuncToDesc_;
extern aclrtAllocatorSetObjToDescFunObj aclrtAllocatorSetObjToDesc_;
extern aclrtAllocatorUnregisterFunObj aclrtAllocatorUnregister_;

void LoadAclAllocatorApiSymbol(const std::string &ascend_path);
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_RT_ALLOCATOR_SYMBOL_H_
