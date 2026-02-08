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
#include "acl_op_symbol.h"
#include <string>
#include "symbol_utils.h"

namespace mindspore::device::ascend {
aclopCreateAttrFunObj aclopCreateAttr_ = nullptr;
aclopSetAttrBoolFunObj aclopSetAttrBool_ = nullptr;
aclopSetAttrDataTypeFunObj aclopSetAttrDataType_ = nullptr;
aclopSetAttrFloatFunObj aclopSetAttrFloat_ = nullptr;
aclopSetAttrIntFunObj aclopSetAttrInt_ = nullptr;
aclopSetAttrListBoolFunObj aclopSetAttrListBool_ = nullptr;
aclopSetAttrListDataTypeFunObj aclopSetAttrListDataType_ = nullptr;
aclopSetAttrListFloatFunObj aclopSetAttrListFloat_ = nullptr;
aclopSetAttrListIntFunObj aclopSetAttrListInt_ = nullptr;
aclopSetAttrListListIntFunObj aclopSetAttrListListInt_ = nullptr;
aclopSetAttrListStringFunObj aclopSetAttrListString_ = nullptr;
aclopSetAttrStringFunObj aclopSetAttrString_ = nullptr;
aclopSetModelDirFunObj aclopSetModelDir_ = nullptr;

void LoadAclOpApiSymbol(const std::string &ascend_path) {
  std::string ascendcl_plugin_path = ascend_path + "lib64/libascendcl.so";
  auto handler = GetLibHandler(ascendcl_plugin_path);
  if (handler == nullptr) {
    MS_LOG(WARNING) << "Dlopen " << ascendcl_plugin_path << " failed!" << dlerror();
    return;
  }
  aclopCreateAttr_ = DlsymAscendFuncObj(aclopCreateAttr, handler);
  aclopSetAttrBool_ = DlsymAscendFuncObj(aclopSetAttrBool, handler);
  aclopSetAttrDataType_ = DlsymAscendFuncObj(aclopSetAttrDataType, handler);
  aclopSetAttrFloat_ = DlsymAscendFuncObj(aclopSetAttrFloat, handler);
  aclopSetAttrInt_ = DlsymAscendFuncObj(aclopSetAttrInt, handler);
  aclopSetAttrListBool_ = DlsymAscendFuncObj(aclopSetAttrListBool, handler);
  aclopSetAttrListDataType_ = DlsymAscendFuncObj(aclopSetAttrListDataType, handler);
  aclopSetAttrListFloat_ = DlsymAscendFuncObj(aclopSetAttrListFloat, handler);
  aclopSetAttrListInt_ = DlsymAscendFuncObj(aclopSetAttrListInt, handler);
  aclopSetAttrListListInt_ = DlsymAscendFuncObj(aclopSetAttrListListInt, handler);
  aclopSetAttrListString_ = DlsymAscendFuncObj(aclopSetAttrListString, handler);
  aclopSetAttrString_ = DlsymAscendFuncObj(aclopSetAttrString, handler);
  aclopSetModelDir_ = DlsymAscendFuncObj(aclopSetModelDir, handler);
  MS_LOG(INFO) << "Load ascend op api success!";
}
}  // namespace mindspore::device::ascend
