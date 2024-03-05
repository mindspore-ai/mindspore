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
#ifndef MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_SYMBOL_H_
#define MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_SYMBOL_H_
#include <string>
#include "acl/acl_rt_allocator.h"
#include "utils/dlopen_macro.h"

namespace mindspore {
namespace transform {

ORIGIN_METHOD(aclInit, aclError, const char *);
ORIGIN_METHOD(aclGetRecentErrMsg, const char *);
ORIGIN_METHOD(aclFinalize, aclError);

extern aclInitFunObj aclInit_;
extern aclGetRecentErrMsgFunObj aclGetRecentErrMsg_;
extern aclFinalizeFunObj aclFinalize_;

void LoadAclApiSymbol(const std::string &ascend_path);
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_SYMBOL_H_
