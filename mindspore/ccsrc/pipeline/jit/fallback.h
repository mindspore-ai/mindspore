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

// Create a PyExecute CNode by old node or debug_info.
CNodePtr CreatePyExecuteCNode(const FuncGraphPtr &fg, const AnfNodePtr &script, const AnfNodePtr &keys,
                              const AnfNodePtr &values, const NodeDebugInfoPtr &debug_info);
CNodePtr CreatePyExecuteCNode(const AnfNodePtr &orig_node, const AnfNodePtr &script, const AnfNodePtr &keys,
                              const AnfNodePtr &values);
CNodePtr CreatePyExecuteCNodeInOrder(const FuncGraphPtr &fg, const AnfNodePtr &script, const AnfNodePtr &keys,
                                     const AnfNodePtr &values, const NodeDebugInfoPtr &debug_info);
CNodePtr CreatePyExecuteCNodeInOrder(const AnfNodePtr &orig_node, const AnfNodePtr &script, const AnfNodePtr &keys,
                                     const AnfNodePtr &values);
void SetPyObjectToLocalVariable(const std::string &key, const py::object &value);
AnfNodePtr ConvertPyObjectToPyExecute(const FuncGraphPtr &fg, const std::string &key, const py::object value,
                                      const AnfNodePtr &node, bool replace);
AnfNodePtr ConvertPyObjectToPyInterpret(const FuncGraphPtr &fg, const std::string &key, const py::object value,
                                        const AnfNodePtr &node, bool replace);
AnfNodePtr ConvertMsClassObjectToPyExecute(const FuncGraphPtr &fg, const ValuePtr &value, const AnfNodePtr &node);

using FormatedVariableTypeFunc = std::function<TypePtr(const std::string &)>;

TypePtr GetJitAnnotationTypeFromComment(const AnfNodePtr &node,
                                        const FormatedVariableTypeFunc &format_type_func = FormatedVariableTypeFunc());

bool ContainsSequenceAnyType(const AbstractBasePtr &abs);

std::string ConvertRealStrToUnicodeStr(const std::string &target, size_t index);
py::object GeneratePyObj(const abstract::AbstractBasePtr &abs);
void AttachListObjToAbs(const AbstractBasePtr &abs, const py::object &obj);

AnfNodePtr ConvertCNodeToPyExecuteForPrim(const CNodePtr &cnode, string name);

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

template <typename T>
bool HasPySeqObject(const std::shared_ptr<T> &owner) {
  if constexpr (std::is_base_of<abstract::AbstractBase, T>()) {
    auto owner_abs_list = dyn_cast<abstract::AbstractList>(owner);
    if (owner_abs_list == nullptr) {
      return false;
    }
    return owner_abs_list->has_list_py_obj();
  }
  constexpr auto py_list_obj_str = "__py_list_object__";
  return owner->has_user_data(py_list_obj_str);
}

template <typename T, typename U>
void SetPySeqObject(const std::shared_ptr<T> &owner, const std::shared_ptr<U> &data) {
  if constexpr (std::is_base_of<abstract::AbstractBase, T>()) {
    auto owner_abs_list = dyn_cast<abstract::AbstractList>(owner);
    if (owner_abs_list == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "Abstract: " << owner->ToString() << " can not attach list object.";
    }
    return owner_abs_list->set_list_py_obj(data);
  }
  constexpr auto py_list_obj_str = "__py_list_object__";
  owner->template set_user_data<U>(py_list_obj_str, data);
}

template <typename T, typename U>
std::shared_ptr<U> GetPySeqObject(const std::shared_ptr<T> &owner) {
  if constexpr (std::is_base_of<abstract::AbstractBase, T>()) {
    auto owner_abs_list = dyn_cast<abstract::AbstractList>(owner);
    if (owner_abs_list == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "Abstract: " << owner->ToString() << " can not get list object.";
    }
    return owner_abs_list->template list_py_obj<U>();
  }
  constexpr auto py_list_obj_str = "__py_list_object__";
  return owner->template user_data<U>(py_list_obj_str);
}

std::string GetPyObjectPtrStr(const py::object &obj);

bool EnableFallbackList();
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

bool HasVariableCondition(const FuncGraphPtr &cur_graph, std::vector<FuncGraphPtr> *prev_graph);
}  // namespace raiseutils
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_FALLBACK_H_
