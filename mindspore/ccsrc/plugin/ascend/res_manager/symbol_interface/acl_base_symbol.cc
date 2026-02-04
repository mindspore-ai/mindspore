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
#include "acl_base_symbol.h"
#include <string>
#include "symbol_utils.h"

namespace mindspore::device::ascend {
aclCreateDataBufferFunObj aclCreateDataBuffer_ = nullptr;
aclCreateTensorDescFunObj aclCreateTensorDesc_ = nullptr;
aclDataTypeSizeFunObj aclDataTypeSize_ = nullptr;
aclDestroyDataBufferFunObj aclDestroyDataBuffer_ = nullptr;
aclDestroyTensorDescFunObj aclDestroyTensorDesc_ = nullptr;
aclGetTensorDescDimFunObj aclGetTensorDescDim_ = nullptr;
aclGetTensorDescDimV2FunObj aclGetTensorDescDimV2_ = nullptr;
aclGetTensorDescNumDimsFunObj aclGetTensorDescNumDims_ = nullptr;
aclSetTensorConstFunObj aclSetTensorConst_ = nullptr;
aclSetTensorDescNameFunObj aclSetTensorDescName_ = nullptr;
aclSetTensorFormatFunObj aclSetTensorFormat_ = nullptr;
aclSetTensorPlaceMentFunObj aclSetTensorPlaceMent_ = nullptr;
aclSetTensorShapeFunObj aclSetTensorShape_ = nullptr;
aclrtGetSocNameFunObj aclrtGetSocName_ = nullptr;
aclUpdateDataBufferFunObj aclUpdateDataBuffer_ = nullptr;
aclGetDataBufferAddrFunObj aclGetDataBufferAddr_ = nullptr;
aclGetTensorDescSizeFunObj aclGetTensorDescSize_ = nullptr;
aclGetRecentErrMsgFunObj aclGetRecentErrMsg_ = nullptr;

void LoadAclBaseApiSymbol(const std::string &ascend_path) {
  std::string aclbase_plugin_path = "lib64/libascendcl.so";
  auto base_handler = GetLibHandler(ascend_path + aclbase_plugin_path);
  if (base_handler == nullptr) {
    MS_LOG(WARNING) << "Dlopen " << aclbase_plugin_path << " failed!" << dlerror();
    return;
  }
  aclCreateDataBuffer_ = DlsymAscendFuncObj(aclCreateDataBuffer, base_handler);
  aclCreateTensorDesc_ = DlsymAscendFuncObj(aclCreateTensorDesc, base_handler);
  aclDataTypeSize_ = DlsymAscendFuncObj(aclDataTypeSize, base_handler);
  aclDestroyDataBuffer_ = DlsymAscendFuncObj(aclDestroyDataBuffer, base_handler);
  aclDestroyTensorDesc_ = DlsymAscendFuncObj(aclDestroyTensorDesc, base_handler);
  aclGetTensorDescDim_ = DlsymAscendFuncObj(aclGetTensorDescDim, base_handler);
  aclGetTensorDescDimV2_ = DlsymAscendFuncObj(aclGetTensorDescDimV2, base_handler);
  aclGetTensorDescNumDims_ = DlsymAscendFuncObj(aclGetTensorDescNumDims, base_handler);
  aclSetTensorConst_ = DlsymAscendFuncObj(aclSetTensorConst, base_handler);
  aclSetTensorDescName_ = DlsymAscendFuncObj(aclSetTensorDescName, base_handler);
  aclSetTensorFormat_ = DlsymAscendFuncObj(aclSetTensorFormat, base_handler);
  aclSetTensorPlaceMent_ = DlsymAscendFuncObj(aclSetTensorPlaceMent, base_handler);
  aclSetTensorShape_ = DlsymAscendFuncObj(aclSetTensorShape, base_handler);
  aclrtGetSocName_ = DlsymAscendFuncObj(aclrtGetSocName, base_handler);
  aclUpdateDataBuffer_ = DlsymAscendFuncObj(aclUpdateDataBuffer, base_handler);
  aclGetDataBufferAddr_ = DlsymAscendFuncObj(aclGetDataBufferAddr, base_handler);
  aclGetTensorDescSize_ = DlsymAscendFuncObj(aclGetTensorDescSize, base_handler);
  aclGetRecentErrMsg_ = DlsymAscendFuncObj(aclGetRecentErrMsg, base_handler);
  MS_LOG(INFO) << "Load acl base api success!";
}

void LoadSimulationAclBaseApi() {
  ASSIGN_SIMU(aclCreateDataBuffer);
  ASSIGN_SIMU(aclCreateTensorDesc);
  ASSIGN_SIMU(aclDataTypeSize);
  ASSIGN_SIMU(aclDestroyDataBuffer);
  ASSIGN_SIMU(aclDestroyTensorDesc);
  ASSIGN_SIMU(aclGetTensorDescDim);
  ASSIGN_SIMU(aclGetTensorDescDimV2);
  ASSIGN_SIMU(aclGetTensorDescNumDims);
  ASSIGN_SIMU(aclSetTensorConst);
  ASSIGN_SIMU(aclSetTensorDescName);
  ASSIGN_SIMU(aclSetTensorFormat);
  ASSIGN_SIMU(aclSetTensorPlaceMent);
  ASSIGN_SIMU(aclSetTensorShape);
  ASSIGN_SIMU(aclUpdateDataBuffer);
  ASSIGN_SIMU(aclrtGetSocName);
  ASSIGN_SIMU(aclGetDataBufferAddr);
  ASSIGN_SIMU(aclGetTensorDescSize);
  ASSIGN_SIMU(aclGetRecentErrMsg);
}
}  // namespace mindspore::device::ascend
