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
#ifndef MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_BASE_SYMBOL_H_
#define MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_BASE_SYMBOL_H_
#include <string>
#include "acl/acl_base.h"
#include "utils/dlopen_macro.h"

namespace mindspore {
namespace transform {

ORIGIN_METHOD(aclCreateDataBuffer, aclDataBuffer *, void *, size_t);
ORIGIN_METHOD(aclCreateTensorDesc, aclTensorDesc *, aclDataType, int, const int64_t *, aclFormat);
ORIGIN_METHOD(aclDataTypeSize, size_t, aclDataType);
ORIGIN_METHOD(aclDestroyDataBuffer, aclError, const aclDataBuffer *);
ORIGIN_METHOD(aclDestroyTensorDesc, void, const aclTensorDesc *);
ORIGIN_METHOD(aclGetTensorDescDimV2, aclError, const aclTensorDesc *, size_t, int64_t *);
ORIGIN_METHOD(aclGetTensorDescNumDims, size_t, const aclTensorDesc *)
ORIGIN_METHOD(aclSetTensorConst, aclError, aclTensorDesc *, void *, size_t)
ORIGIN_METHOD(aclSetTensorDescName, void, aclTensorDesc *, const char *)
ORIGIN_METHOD(aclSetTensorFormat, aclError, aclTensorDesc *, aclFormat)
ORIGIN_METHOD(aclSetTensorPlaceMent, aclError, aclTensorDesc *, aclMemType)
ORIGIN_METHOD(aclSetTensorShape, aclError, aclTensorDesc *, int, const int64_t *)
ORIGIN_METHOD(aclrtGetSocName, const char *)

extern aclCreateDataBufferFunObj aclCreateDataBuffer_;
extern aclCreateTensorDescFunObj aclCreateTensorDesc_;
extern aclDataTypeSizeFunObj aclDataTypeSize_;
extern aclDestroyDataBufferFunObj aclDestroyDataBuffer_;
extern aclDestroyTensorDescFunObj aclDestroyTensorDesc_;
extern aclGetTensorDescDimV2FunObj aclGetTensorDescDimV2_;
extern aclGetTensorDescNumDimsFunObj aclGetTensorDescNumDims_;
extern aclSetTensorConstFunObj aclSetTensorConst_;
extern aclSetTensorDescNameFunObj aclSetTensorDescName_;
extern aclSetTensorFormatFunObj aclSetTensorFormat_;
extern aclSetTensorPlaceMentFunObj aclSetTensorPlaceMent_;
extern aclSetTensorShapeFunObj aclSetTensorShape_;
extern aclrtGetSocNameFunObj aclrtGetSocName_;

void LoadAclBaseApiSymbol(const std::string &ascend_path);
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_BASE_SYMBOL_H_
