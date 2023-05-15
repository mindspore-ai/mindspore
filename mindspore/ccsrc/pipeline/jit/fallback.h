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

#include <memory>
#include <string>
#include <vector>

#include "ir/anf.h"
#include "ir/dtype/type.h"
#include "include/common/utils/python_adapter.h"
#include "pipeline/jit/parse/resolve.h"

namespace mindspore {
namespace fallback {
constexpr auto kPyExecPrefix = "__py_exec_index";
constexpr auto kPyExecSuffix = "__";
constexpr auto kUnderLine = "_";
constexpr auto kHexPrefix = "0x";
constexpr auto kPyExecuteSlice = "[__start__:__stop__:__step__]";

AnfNodePtr GeneratePyExecuteNodeWithScriptSrc(const FuncGraphPtr &func_graph, const TypePtrList &types,
                                              const AnfNodePtrList &node_inputs, std::string script_str);
void SetNodeExprSrc(const AnfNodePtr &node, const std::string &expr_src);
std::string GetNodeExprSrc(const AnfNodePtr &node);
std::string GeneratePyExecuteScriptForBinOrComp(const std::string &left, const std::string &right,
                                                const std::string &op);
std::string GeneratePyExecuteScriptForUnary(const std::string &operand, const std::string &op);
std::string GeneratePyExecuteScriptForSubscript(const std::string &value, const std::string &slice, bool is_slice);

AnfNodePtr ConvertPyObjectToPyExecute(const FuncGraphPtr &fg, const std::string &key, const py::object value,
                                      const AnfNodePtr &node);
AnfNodePtr ConvertInterpretedObjectToPyExecute(const FuncGraphPtr &fg, const ValuePtr &value, const AnfNodePtr &node);

using FormatedVariableTypeFunc = std::function<TypePtr(const std::string &)>;

TypePtr GetJitAnnotationTypeFromComment(const AnfNodePtr &node,
                                        const FormatedVariableTypeFunc &format_type_func = FormatedVariableTypeFunc());

template <typename T>
bool HasRealType(const std::shared_ptr<T> &owner) {
  return owner->has_user_data("__py_execute_real_type__");
}

template <typename T, typename U>
void SetRealType(const std::shared_ptr<T> &owner, const std::shared_ptr<U> &data) {
  owner->template set_user_data<U>("__py_execute_real_type__", data);
}

template <typename T, typename U>
std::shared_ptr<U> GetRealType(const std::shared_ptr<T> &owner) {
  return owner->template user_data<U>("__py_execute_real_type__");
}

template <typename T>
bool HasRealShape(const std::shared_ptr<T> &owner) {
  return owner->has_user_data("__py_execute_real_shape__");
}

template <typename T, typename U>
void SetRealShape(const std::shared_ptr<T> &owner, const std::shared_ptr<U> &data) {
  owner->template set_user_data<U>("__py_execute_real_shape__", data);
}

template <typename T, typename U>
std::shared_ptr<U> GetRealShape(const std::shared_ptr<T> &owner) {
  return owner->template user_data<U>("__py_execute_real_shape__");
}
}  // namespace fallback

namespace raiseutils {
using ClassTypePtr = std::shared_ptr<parse::ClassType>;

struct KeyValueInfo {
  int num_str = 0;
  std::vector<AnfNodePtr> keys;
  std::vector<AnfNodePtr> values;
};

std::string GetScalarStringValue(const AbstractBasePtr &abs);

std::string GetExceptionType(const AbstractBasePtr &abs, const AnfNodePtr &cnode,
                             const std::shared_ptr<KeyValueInfo> &key_value, bool has_variable = true);

std::string GetTupleOrListString(const AbstractBasePtr &arg, const AnfNodePtr &input,
                                 const std::shared_ptr<KeyValueInfo> &key_value, bool need_symbol = false,
                                 bool need_comma = false);

bool CheckHasVariable(const AbstractBasePtr &arg);

std::string GetVariable(const AnfNodePtr &input, const bool need_symbol, const std::shared_ptr<KeyValueInfo> &key_value,
                        std::string exception_str);

std::string GetExceptionString(const AbstractBasePtr &arg, const AnfNodePtr &input,
                               const std::shared_ptr<KeyValueInfo> &key_value, bool need_symbol = false,
                               bool need_comma = false);

bool CheckNeedSymbol(const AbstractBasePtr &abs);

bool CheckIsStr(const AbstractBasePtr &abs);

std::string MakeRaiseKey(const int index);

bool HasVariableCondition(FuncGraphPtr cur_graph, FuncGraphPtr prev_graph = nullptr);
}  // namespace raiseutils
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_FALLBACK_H_
