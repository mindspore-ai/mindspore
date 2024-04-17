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
#ifndef MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_COMPILER_SYMBOL_H_
#define MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_COMPILER_SYMBOL_H_
#include <string>
#include "acl/acl_op_compiler.h"
#include "utils/dlopen_macro.h"

namespace mindspore {
namespace transform {

ORIGIN_METHOD(aclopCompileAndExecute, aclError, const char *, int, const aclTensorDesc *const[],
              const aclDataBuffer *const[], int, const aclTensorDesc *const[], aclDataBuffer *const[],
              const aclopAttr *, aclopEngineType, aclopCompileType, const char *, aclrtStream);
ORIGIN_METHOD(aclopCompileAndExecuteV2, aclError, const char *, int, aclTensorDesc *[], aclDataBuffer *[], int,
              aclTensorDesc *[], aclDataBuffer *[], aclopAttr *, aclopEngineType, aclopCompileType, const char *,
              aclrtStream);
ORIGIN_METHOD(aclSetCompileopt, aclError, aclCompileOpt, const char *);
ORIGIN_METHOD(aclopSetCompileFlag, aclError, aclOpCompileFlag);
ORIGIN_METHOD(aclGenGraphAndDumpForOp, aclError, const char *, int, const aclTensorDesc *const[],
              const aclDataBuffer *const[], int, const aclTensorDesc *const[], aclDataBuffer *const[],
              const aclopAttr *, aclopEngineType, const char *, const aclGraphDumpOption *);

extern aclopCompileAndExecuteFunObj aclopCompileAndExecute_;
extern aclopCompileAndExecuteV2FunObj aclopCompileAndExecuteV2_;
extern aclSetCompileoptFunObj aclSetCompileopt_;
extern aclopSetCompileFlagFunObj aclopSetCompileFlag_;
extern aclGenGraphAndDumpForOpFunObj aclGenGraphAndDumpForOp_;

void LoadAclOpCompilerApiSymbol(const std::string &ascend_path);
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_COMPILER_SYMBOL_H_
