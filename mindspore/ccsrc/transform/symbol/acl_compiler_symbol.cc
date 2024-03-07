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
#include "transform/symbol/acl_compiler_symbol.h"

namespace mindspore {
namespace transform {
aclopCompileAndExecuteFunObj aclopCompileAndExecute_ = nullptr;
aclopCompileAndExecuteV2FunObj aclopCompileAndExecuteV2_ = nullptr;
aclSetCompileoptFunObj aclSetCompileopt_ = nullptr;
aclopSetCompileFlagFunObj aclopSetCompileFlag_ = nullptr;
aclGenGraphAndDumpForOpFunObj aclGenGraphAndDumpForOp_ = nullptr;

void LoadAclOpCompilerApiSymbol(const std::string &ascend_path) {
  std::string complier_plugin_path = "lib64/libacl_op_compiler.so";
  auto handler = GetLibHandler(ascend_path + complier_plugin_path);
  if (handler == nullptr) {
    MS_LOG(EXCEPTION) << "Dlopen " << complier_plugin_path << " failed!" << dlerror();
  }
  aclopCompileAndExecute_ = DlsymAscendFuncObj(aclopCompileAndExecute, handler);
  aclopCompileAndExecuteV2_ = DlsymAscendFuncObj(aclopCompileAndExecuteV2, handler);
  aclSetCompileopt_ = DlsymAscendFuncObj(aclSetCompileopt, handler);
  aclopSetCompileFlag_ = DlsymAscendFuncObj(aclopSetCompileFlag, handler);
  aclGenGraphAndDumpForOp_ = DlsymAscendFuncObj(aclGenGraphAndDumpForOp, handler);
  MS_LOG(INFO) << "Load acl op compiler api success!";
}

}  // namespace transform
}  // namespace mindspore
