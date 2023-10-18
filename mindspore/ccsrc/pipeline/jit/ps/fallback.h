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
#include "abstract/abstract_value.h"
#include "include/common/utils/python_adapter.h"
#include "pipeline/jit/ps/parse/resolve.h"

namespace mindspore {
namespace fallback {
constexpr auto kPyExecPrefix = "__py_exec_index";
constexpr auto kPyExecSuffix = "__";
constexpr auto kUnderLine = "_";
constexpr auto kHexPrefix = "0x";
constexpr auto kPyExecuteSlice = "[__start__:__stop__:__step__]";

AnfNodePtr GeneratePyInterpretNodeWithScriptSrc(const FuncGraphPtr &func_graph, const TypePtrList &types,
                                                const AnfNodePtrList &node_inputs, std::string script_str);
void SetNodeExprSrc(const AnfNodePtr &node, const std::string &expr_src);
std::string GetNodeExprSrc(const AnfNodePtr &node);
std::string GeneratePyInterpretScriptForBinOrComp(const std::string &left, const std::string &right,
                                                  const std::string &op);
std::string GeneratePyInterpretScriptForUnary(const std::string &operand, const std::string &op);
std::string GeneratePyInterpretScriptForSubscript(const std::string &value, const std::string &slice, bool is_slice);
std::string GeneratePyInterpretScriptForCallNode(const AnfNodePtr &call_node, const std::string &name_id);

// Create a PyExecute CNode by old node or debug_info.
CNodePtr CreatePyExecuteCNode(const FuncGraphPtr &fg, const AnfNodePtr &script, const AnfNodePtr &keys,
                              const AnfNodePtr &values, const NodeDebugInfoPtr &debug_info);
CNodePtr CreatePyExecuteCNode(const AnfNodePtr &orig_node, const AnfNodePtr &script, const AnfNodePtr &keys,
                              const AnfNodePtr &values);
CNodePtr CreatePyExecuteCNodeInOrder(const FuncGraphPtr &fg, const AnfNodePtr &script, const AnfNodePtr &keys,
                                     const AnfNodePtr &values, const NodeDebugInfoPtr &debug_info);
CNodePtr CreatePyExecuteCNodeInOrder(const AnfNodePtr &orig_node, const AnfNodePtr &script, const AnfNodePtr &keys,
                                     const AnfNodePtr &values);
// Create a PyInterpret CNode by old node or debug_info.
CNodePtr CreatePyInterpretCNodeInOrder(const FuncGraphPtr &fg, const std::string &script_text,
                                       const py::object &global_dict_obj, const AnfNodePtr &local_dict_node,
                                       const NodeDebugInfoPtr &debug_info);
void SetPyObjectToLocalVariable(const std::string &key, const py::object &value);
AnfNodePtr ConvertPyObjectToPyExecute(const FuncGraphPtr &fg, const std::string &key, const py::object value,
                                      const AnfNodePtr &node, bool replace);
AnfNodePtr ConvertPyObjectToPyInterpret(const FuncGraphPtr &fg, const std::string &key, const py::object value,
                                        const AnfNodePtr &node, bool replace);
AnfNodePtr ConvertMsClassObjectToPyExecute(const FuncGraphPtr &fg, const ValuePtr &value, const AnfNodePtr &node);

using FormatedVariableTypeFunc = std::function<TypePtr(const std::string &)>;

TypePtr GetJitAnnotationTypeFromComment(const AnfNodePtr &node,
                                        const FormatedVariableTypeFunc &format_type_func = FormatedVariableTypeFunc());
bool GetJitAnnotationSideEffectFromComment(const AnfNodePtr &node);
bool ContainsSequenceAnyType(const AbstractBasePtr &abs);

std::string ConvertRealStrToUnicodeStr(const std::string &target, size_t index);

std::string GetPyObjectPtrStr(const py::object &obj);
bool EnableFallbackListDictInplace();

// Generate python object according to abstract.
py::object GeneratePyObj(const abstract::AbstractBasePtr &abs);
// Handle python object for abstract using ExtraInfoHolder.
void AttachPyObjToExtraInfoHolder(const abstract::AbstractBasePtr &abs, const py::object &obj, bool create_in_graph);
py::object GetObjFromExtraInfoHolder(const abstract::AbstractBasePtr &abs);
bool GetCreateInGraphFromExtraInfoHolder(const abstract::AbstractBasePtr &abs);
bool HasObjInExtraInfoHolder(const abstract::AbstractBasePtr &abs);
// Attach python object to abstract recursively using ExtraInfoHolder.
void AttachPyObjToAbs(const AbstractBasePtr &abs, const py::object &obj, bool create_in_graph);
// Handle python object for AnfNode.
void SetPyObjectToNode(const AnfNodePtr &node, const py::object &obj);
bool HasPyObjectInNode(const AnfNodePtr &node);
py::object GetPyObjectFromNode(const AnfNodePtr &node);

AnfNodePtr ConvertCNodeToPyExecuteForPrim(const CNodePtr &cnode, const string &name);

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

AnfNodePtr GenerateOnesOrZerosLikeNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input,
                                       const std::string &type);
}  // namespace fallback

namespace raiseutils {
using ClassTypePtr = std::shared_ptr<parse::ClassType>;

struct KeyValueInfo {
  int num_str = 0;
  std::vector<AnfNodePtr> keys;
  std::vector<AnfNodePtr> values;
};

std::string GetExceptionType(const AbstractBasePtr &abs, const AnfNodePtr &cnode,
                             const std::shared_ptr<KeyValueInfo> &key_value, bool has_variable = true);

bool CheckHasVariable(const AbstractBasePtr &arg);

std::string GetExceptionString(const AbstractBasePtr &arg, const AnfNodePtr &input,
                               const std::shared_ptr<KeyValueInfo> &key_value, bool need_symbol = false,
                               bool need_comma = false);

bool CheckNeedSymbol(const AbstractBasePtr &abs);

std::string MakeRaiseKey(int index);

bool HasVariableCondition(const FuncGraphPtr &cur_graph);
}  // namespace raiseutils
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_FALLBACK_H_
