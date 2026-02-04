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
#ifndef MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_OP_SYMBOL_H_
#define MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_OP_SYMBOL_H_
#include <string>
#include "acl/acl_op.h"
#include "utils/dlopen_macro.h"

namespace mindspore::device::ascend {

ORIGIN_METHOD(aclopCreateAttr, aclopAttr *)
ORIGIN_METHOD(aclopSetAttrBool, aclError, aclopAttr *, const char *, uint8_t)
ORIGIN_METHOD(aclopSetAttrDataType, aclError, aclopAttr *, const char *, aclDataType)
ORIGIN_METHOD(aclopSetAttrFloat, aclError, aclopAttr *, const char *, float)
ORIGIN_METHOD(aclopSetAttrInt, aclError, aclopAttr *, const char *, int64_t)
ORIGIN_METHOD(aclopSetAttrListBool, aclError, aclopAttr *, const char *, int, const uint8_t *)
ORIGIN_METHOD(aclopSetAttrListDataType, aclError, aclopAttr *, const char *, int, const aclDataType[])
ORIGIN_METHOD(aclopSetAttrListFloat, aclError, aclopAttr *, const char *, int, const float *)
ORIGIN_METHOD(aclopSetAttrListInt, aclError, aclopAttr *, const char *, int, const int64_t *)
ORIGIN_METHOD(aclopSetAttrListListInt, aclError, aclopAttr *, const char *, int, const int *, const int64_t *const[])
ORIGIN_METHOD(aclopSetAttrListString, aclError, aclopAttr *, const char *, int, const char **)
ORIGIN_METHOD(aclopSetAttrString, aclError, aclopAttr *, const char *, const char *)
ORIGIN_METHOD(aclopSetModelDir, aclError, const char *)

extern aclopCreateAttrFunObj aclopCreateAttr_;
extern aclopSetAttrBoolFunObj aclopSetAttrBool_;
extern aclopSetAttrDataTypeFunObj aclopSetAttrDataType_;
extern aclopSetAttrFloatFunObj aclopSetAttrFloat_;
extern aclopSetAttrIntFunObj aclopSetAttrInt_;
extern aclopSetAttrListBoolFunObj aclopSetAttrListBool_;
extern aclopSetAttrListDataTypeFunObj aclopSetAttrListDataType_;
extern aclopSetAttrListFloatFunObj aclopSetAttrListFloat_;
extern aclopSetAttrListIntFunObj aclopSetAttrListInt_;
extern aclopSetAttrListListIntFunObj aclopSetAttrListListInt_;
extern aclopSetAttrListStringFunObj aclopSetAttrListString_;
extern aclopSetAttrStringFunObj aclopSetAttrString_;
extern aclopSetModelDirFunObj aclopSetModelDir_;

void LoadAclOpApiSymbol(const std::string &ascend_path);
}  // namespace mindspore::device::ascend

#endif  // MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_OP_SYMBOL_H_
