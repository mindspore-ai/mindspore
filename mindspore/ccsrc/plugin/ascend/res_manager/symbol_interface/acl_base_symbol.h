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

namespace mindspore::device::ascend {
ORIGIN_METHOD_WITH_SIMU(aclCreateDataBuffer, aclDataBuffer *, void *, size_t);
ORIGIN_METHOD_WITH_SIMU(aclCreateTensorDesc, aclTensorDesc *, aclDataType, int, const int64_t *, aclFormat);
ORIGIN_METHOD_WITH_SIMU(aclDataTypeSize, size_t, aclDataType);
ORIGIN_METHOD_WITH_SIMU(aclDestroyDataBuffer, aclError, const aclDataBuffer *);
ORIGIN_METHOD_WITH_SIMU(aclDestroyTensorDesc, void, const aclTensorDesc *);
ORIGIN_METHOD_WITH_SIMU(aclGetTensorDescDim, int64_t, const aclTensorDesc *, size_t);
ORIGIN_METHOD_WITH_SIMU(aclGetTensorDescDimV2, aclError, const aclTensorDesc *, size_t, int64_t *);
ORIGIN_METHOD_WITH_SIMU(aclGetTensorDescNumDims, size_t, const aclTensorDesc *)
ORIGIN_METHOD_WITH_SIMU(aclSetTensorConst, aclError, aclTensorDesc *, void *, size_t)
ORIGIN_METHOD_WITH_SIMU(aclSetTensorDescName, void, aclTensorDesc *, const char *)
ORIGIN_METHOD_WITH_SIMU(aclSetTensorFormat, aclError, aclTensorDesc *, aclFormat)
ORIGIN_METHOD_WITH_SIMU(aclSetTensorPlaceMent, aclError, aclTensorDesc *, aclMemType)
ORIGIN_METHOD_WITH_SIMU(aclSetTensorShape, aclError, aclTensorDesc *, int, const int64_t *)
ACLRT_GET_SOC_NAME_WITH_SIMU(aclrtGetSocName, const char *)
ORIGIN_METHOD_WITH_SIMU(aclUpdateDataBuffer, aclError, aclDataBuffer *, void *, size_t)
ORIGIN_METHOD_WITH_SIMU(aclGetDataBufferAddr, void *, const aclDataBuffer *)
ORIGIN_METHOD_WITH_SIMU(aclGetTensorDescSize, size_t, const aclTensorDesc *)
ORIGIN_METHOD_WITH_SIMU(aclGetRecentErrMsg, const char *)

void LoadAclBaseApiSymbol(const std::string &ascend_path);
void LoadSimulationAclBaseApi();
}  // namespace mindspore::device::ascend

#endif  // MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_BASE_SYMBOL_H_
