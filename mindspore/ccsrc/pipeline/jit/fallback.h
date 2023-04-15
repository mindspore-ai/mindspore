/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_FALLBACK_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_FALLBACK_H_

#include <string>
#include "ir/anf.h"
#include "ir/dtype/type.h"

namespace mindspore {
constexpr auto kPyExecPrefix = "__py_exec_index";
constexpr auto kPyExecSuffix = "__";
constexpr auto kUnderLine = "_";
constexpr auto kHexPrefix = "0x";
constexpr auto kPyExecuteSlice = "[__start__:__stop__:__step__]";

AnfNodePtr GeneratePyExecuteNodeWithScriptSrc(const FuncGraphPtr &func_graph, const TypePtrList &types,
                                              const AnfNodePtrList &node_inputs, std::string script_str);
void SetNodeExprSrc(const AnfNodePtr &node, const std::string &expr_src);
std::string GetNodeExprSrc(const AnfNodePtr &node);
std::string ConvertRealStrToUnicodeStr(const std::string &target, size_t index);
std::string ConvertToRealStr(const std::string &target);
std::string GeneratePyExecuteScriptForBinOrComp(const std::string &left, const std::string &right,
                                                const std::string &op);
std::string GeneratePyExecuteScriptForUnary(const std::string &operand, const std::string &op);
std::string GeneratePyExecuteScriptForSubscript(const std::string &value, const std::string &slice, bool is_slice);

AnfNodePtr ConvertInterpretedObjectToPyExecute(const FuncGraphPtr &fg, const ValuePtr &value, const AnfNodePtr &node);
TypePtr GetJitAnnotationTypeFromComment(const AnfNodePtr &node);
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_FALLBACK_H_
