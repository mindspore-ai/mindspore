/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#include "frontend/jit/ps/static_analysis/prim_utils.h"

#include <algorithm>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>

#include "frontend/operator/composite/do_signature.h"
#include "frontend/operator/ops.h"
#include "frontend/operator/composite/functional_overload.h"
#include "include/utils/operator/primitive_utils.h"
#include "ir/core_ops_primitive.h"
#include "frontend/jit/ps/fallback.h"
#include "frontend/jit/ps/parse/data_converter.h"
#include "frontend/jit/ps/parse/parse_base.h"
#include "utils/flags.h"
#include "utils/log_adapter.h"
#include "ir/dtype/tensor_type.h"

namespace mindspore {
namespace abstract {
using OP_DTYPE = mindspore::ops::OP_DTYPE;

namespace {
template <typename T>
bool ValidateSequenceType(const AbstractBasePtr &abs_seq, OP_DTYPE type_elem) {
  if (!abs_seq->isa<T>()) {
    return false;
  }
  if (type_elem == OP_DTYPE::DT_ANY) {
    return true;
  }
  auto abs = abs_seq->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->dynamic_len()) {
    return true;
  }
  for (const auto &abs_elem : abs->elements()) {
    if (!ValidateArgsType(abs_elem, type_elem)) {
      return false;
    }
  }
  return true;
}

bool ValidateArgsSequenceType(const AbstractBasePtr &abs_arg, OP_DTYPE type_arg) {
  switch (static_cast<int>(type_arg)) {
    case OP_DTYPE::DT_TUPLE_BOOL: {
      return ValidateSequenceType<abstract::AbstractTuple>(abs_arg, OP_DTYPE::DT_BOOL);
    }
    case OP_DTYPE::DT_TUPLE_INT: {
      return ValidateSequenceType<abstract::AbstractTuple>(abs_arg, OP_DTYPE::DT_INT);
    }
    case OP_DTYPE::DT_TUPLE_FLOAT: {
      return ValidateSequenceType<abstract::AbstractTuple>(abs_arg, OP_DTYPE::DT_FLOAT);
    }
    case OP_DTYPE::DT_TUPLE_NUMBER: {
      return ValidateSequenceType<abstract::AbstractTuple>(abs_arg, OP_DTYPE::DT_NUMBER);
    }
    case OP_DTYPE::DT_TUPLE_TENSOR: {
      return ValidateSequenceType<abstract::AbstractTuple>(abs_arg, OP_DTYPE::DT_TENSOR);
    }
    case OP_DTYPE::DT_TUPLE_STR: {
      return ValidateSequenceType<abstract::AbstractTuple>(abs_arg, OP_DTYPE::DT_STR);
    }
    case OP_DTYPE::DT_TUPLE_ANY: {
      return ValidateSequenceType<abstract::AbstractTuple>(abs_arg, OP_DTYPE::DT_ANY);
    }
    case OP_DTYPE::DT_LIST_BOOL: {
      return ValidateSequenceType<abstract::AbstractList>(abs_arg, OP_DTYPE::DT_BOOL);
    }
    case OP_DTYPE::DT_LIST_INT: {
      return ValidateSequenceType<abstract::AbstractList>(abs_arg, OP_DTYPE::DT_INT);
    }
    case OP_DTYPE::DT_LIST_FLOAT: {
      return ValidateSequenceType<abstract::AbstractList>(abs_arg, OP_DTYPE::DT_FLOAT);
    }
    case OP_DTYPE::DT_LIST_NUMBER: {
      return ValidateSequenceType<abstract::AbstractList>(abs_arg, OP_DTYPE::DT_NUMBER);
    }
    case OP_DTYPE::DT_LIST_TENSOR: {
      return ValidateSequenceType<abstract::AbstractList>(abs_arg, OP_DTYPE::DT_TENSOR);
    }
    case OP_DTYPE::DT_LIST_STR: {
      return ValidateSequenceType<abstract::AbstractList>(abs_arg, OP_DTYPE::DT_STR);
    }
    case OP_DTYPE::DT_LIST_ANY: {
      return ValidateSequenceType<abstract::AbstractList>(abs_arg, OP_DTYPE::DT_ANY);
    }
    default: {
      MS_EXCEPTION(ValueError) << "Unknown op dtype " << prim::OpDTypeToString(type_arg);
    }
  }
}
}  // namespace

bool ValidateArgsType(const AbstractBasePtr &abs_arg, OP_DTYPE type_arg) {
  auto abs_type = abs_arg->BuildType();
  MS_EXCEPTION_IF_NULL(abs_type);
  switch (static_cast<int>(type_arg)) {
    case OP_DTYPE::DT_ANY: {
      return true;
    }
    case OP_DTYPE::DT_BOOL: {
      return abs_arg->isa<abstract::AbstractScalar>() && abs_type->isa<Bool>();
    }
    case OP_DTYPE::DT_INT: {
      return abs_arg->isa<abstract::AbstractScalar>() && (abs_type->isa<Int>() || abs_type->isa<UInt>());
    }
    case OP_DTYPE::DT_FLOAT: {
      return abs_arg->isa<abstract::AbstractScalar>() && (abs_type->isa<Float>() || abs_type->isa<BFloat>());
    }
    case OP_DTYPE::DT_NUMBER: {
      return abs_arg->isa<abstract::AbstractScalar>() && abs_type->isa<Number>();
    }
    case OP_DTYPE::DT_STR: {
      return abs_arg->isa<abstract::AbstractScalar>() && abs_type->isa<String>();
    }
    case OP_DTYPE::DT_TENSOR: {
      return abs_arg->isa<abstract::AbstractTensor>();
    }
    case OP_DTYPE::DT_TYPE: {
      return abs_arg->isa<abstract::AbstractType>() && abs_type->isa<Type>();
    }
    default: {
      return ValidateArgsSequenceType(abs_arg, type_arg);
    }
  }
  return false;
}

AnfNodePtrList GetPrimitiveInitArgs(const PrimitivePyPtr &prim_py, const ops::OpDef *op_def) {
  MS_EXCEPTION_IF_NULL(prim_py);
  MS_EXCEPTION_IF_NULL(op_def);

  std::vector<AnfNodePtr> prim_init_arg_nodes;
  auto obj = prim_py->GetPyObj();

  for (const auto &op_arg : op_def->args_) {
    if (op_arg.as_init_arg_) {
      auto arg_name = op_arg.arg_name_;
      py::object arg_value = py::getattr(obj, common::SafeCStr(arg_name));
      ValuePtr converted_ret = nullptr;
      bool converted = parse::ConvertData(arg_value, &converted_ret);
      if (!converted) {
        MS_LOG(INTERNAL_EXCEPTION) << "Cannot convert initialization arg: (" << arg_name << ": " << py::str(arg_value)
                                   << ") in Primitive '" << prim_py->name() << "'.";
      }
      (void)prim_init_arg_nodes.emplace_back(NewValueNode(converted_ret));
    }
  }
  MS_LOG(DEBUG) << "PrimitivePy " << prim_py->name() << " has " << prim_init_arg_nodes.size() << " __init__() args";
  return prim_init_arg_nodes;
}

bool ValidateArgOptional(const AbstractBasePtr &abs_arg, const ops::OpInputArg &input_arg) {
  if (!input_arg.is_optional_) {
    return false;
  }

  auto abs_type = abs_arg->BuildType();
  MS_EXCEPTION_IF_NULL(abs_type);
  return abs_type->isa<TypeNone>();
}

bool ValidateArgSpecialType(const std::string &op_name, const AbstractBasePtr &abs, const ops::OpInputArg &op_arg) {
  if (abs->isa<abstract::AbstractKeywordArg>()) {
    MS_EXCEPTION(TypeError) << "For Primitive[" << op_name
                            << "], only positional arguments as inputs are supported, but got " << abs->ToString();
  }
  return fallback::ContainsSequenceAnyType(abs) || ValidateArgOptional(abs, op_arg) ||
         abstract::ValidateArgsType(abs, op_arg.arg_dtype_);
}

void GetKeywordArgsMap(const AbstractBasePtr &input_abs, const std::vector<ops::OpInputArg> &op_args,
                       const AnfNodePtr &input, const FuncGraphPtr &graph, std::map<std::string, AnfNodePtr> *key_map) {
  auto input_kwarg_abs = input_abs->cast<AbstractKeywordArgPtr>();
  const auto &key = input_kwarg_abs->get_key();
  bool is_key_valid = std::any_of(op_args.begin(), op_args.end(),
                                  [&key](const ops::OpInputArg &op_arg) { return key == op_arg.arg_name_; });
  if (is_key_valid) {
    const auto &kwarg_value = graph->NewCNode({NewValueNode(prim::kPrimExtractKeywordArg), NewValueNode(key), input});
    (*key_map)[key] = kwarg_value;
  } else {
    MS_LOG(EXCEPTION) << "Got an unexpected keyword argument '" << key << "'.";
  }
}

AnfNodePtrList GeneratePrimitiveDefaultArgs(const std::string &op_name, const std::vector<AnfNodePtr> &args_list,
                                            const std::vector<ops::OpInputArg> &op_args,
                                            const std::function<AbstractBasePtr(const AnfNodePtr &)> &eval_func,
                                            const FuncGraphPtr &graph) {
  size_t args_size = args_list.size();
  AnfNodePtrList nodes;
  std::map<std::string, AnfNodePtr> key_map;
  for (size_t idx = 0; idx < args_list.size(); ++idx) {
    auto input = args_list[idx];
    if (IsMonad(input)) {
      --args_size;
      continue;
    }
    auto input_abs = eval_func(input);
    if (input_abs->isa<AbstractKeywordArg>()) {
      GetKeywordArgsMap(input_abs, op_args, input, graph, &key_map);
    } else {
      (void)nodes.emplace_back(input);
      continue;
    }
  }
  args_size -= key_map.size();
  if (args_size < op_args.size()) {
    for (size_t i = args_size; i < op_args.size(); ++i) {
      auto arg_name = op_args[i].arg_name_;
      auto iter = key_map.find(arg_name);
      if (iter != key_map.end()) {
        MS_LOG(DEBUG) << "Get args for Primitive[" << op_name << "]: " << iter->second->DebugString();
        (void)nodes.emplace_back(iter->second);
        (void)key_map.erase(arg_name);
      } else {
        auto default_arg = parse::GetArgDefaultValue(op_name, arg_name);
        if (default_arg == nullptr) {
          break;
        }
        MS_LOG(DEBUG) << "Get the default value of '" << arg_name << "' attribute of Primitive[" << op_name
                      << "], which is " << default_arg->ToString() << ".";
        (void)nodes.emplace_back(NewValueNode(default_arg));
      }
    }
  }

  if (nodes.size() != op_args.size()) {
    std::string args_type_str = (op_args.size() != 0 && op_args[0].as_init_arg_) ? "init arguments" : "inputs";
    MS_EXCEPTION(TypeError) << "For Operator[" << op_name << "], the number of " << args_type_str
                            << " (including default arguments) should be " << op_args.size()
                            << ", but the actual number of inputs is not satisfied, which is " << args_size << ".";
  }
  return nodes;
}

namespace {

inline int64_t OpDtypeToInt(ops::OP_DTYPE dtype) { return static_cast<int64_t>(dtype); }

AnfNodePtr GetNodeAfterTypeConversion(const AnfNodePtr &node, const ops::OpInputArg &op_arg, const FuncGraphPtr &fg,
                                      const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(fg);
  // If src_cast_dtype is empty, do no need to do type conversion.
  if (op_arg.cast_dtype_.empty()) {
    return node;
  }
  const auto convert_func =
    prim::GetPythonOps(parse::PYTHON_MOD_PRIMITIVE_OP_TYPE_CAST, parse::PYTHON_MOD_PRIMITIVE_ARG_DTYPE_CAST_MODULE);
  auto convert_fg = dyn_cast<FuncGraph>(convert_func);
  MS_EXCEPTION_IF_NULL(convert_fg);
  convert_fg->set_manager(fg->manager());
  auto res = fg->NewCNodeInOrder(
    {NewValueNode(convert_fg), node, NewValueNode(OpDtypeToInt(op_arg.arg_dtype_)), NewValueNode(op_name)});
  res->set_debug_info(node->debug_info());
  return res;
}

bool ValidateAndConvertArgsType(const std::string &op_name, const std::vector<ops::OpInputArg> &op_args,
                                const AbstractBasePtrList &abs_list, const FuncGraphPtr &fg,
                                std::vector<AnfNodePtr> *nodes) {
  bool exist_undetermined_arg = false;
  for (size_t i = 0; i < op_args.size(); ++i) {
    auto op_arg = op_args[i];
    auto abs_arg = abs_list[i];
    if (HasAbstractType<AbstractUndetermined>(abs_arg)) {
      exist_undetermined_arg = true;
    }
    if (ValidateArgSpecialType(op_name, abs_arg, op_arg)) {
      continue;
    }
    bool match = false;
    auto cast_dtypes = op_arg.cast_dtype_;
    for (size_t j = 0; j < cast_dtypes.size(); ++j) {
      if (abstract::ValidateArgsType(abs_arg, cast_dtypes[j])) {
        (*nodes)[i] = GetNodeAfterTypeConversion((*nodes)[i], op_arg, fg, op_name);
        match = true;
        break;
      }
    }
    if (!match && !exist_undetermined_arg) {
      return false;
    }
  }
  return true;
}

AnfNodePtr GetNodeAfterArgHandler(const AnfNodePtr &node, const std::string &op_name, const ops::OpInputArg &op_arg,
                                  const AbstractBasePtr &abs, const FuncGraphPtr &fg) {
  if (op_arg.arg_handler_.empty()) {
    return node;
  }
  if (op_arg.is_optional_ && abs->isa<AbstractNone>()) {
    return node;
  }
  const auto arg_handler_func = prim::GetPythonOps(op_arg.arg_handler_, parse::PYTHON_MOD_PRIMITIVE_ARG_HANDLER_MODULE);
  MS_LOG(DEBUG) << "The arg handler function for '" << op_arg.arg_name_ << "' of Primitive[" << op_name << "] is "
                << arg_handler_func->ToString() << ".";
  if (arg_handler_func->isa<Primitive>()) {
    auto arg_handler_fg = dyn_cast<Primitive>(arg_handler_func);
    MS_EXCEPTION_IF_NULL(arg_handler_fg);
    auto res =
      fg->NewCNodeInOrder({NewValueNode(arg_handler_fg), NewValueNode(op_name), NewValueNode(op_arg.arg_name_), node});
    res->set_debug_info(node->debug_info());
    return res;
  }
  auto arg_handler_fg = dyn_cast<FuncGraph>(arg_handler_func);
  MS_EXCEPTION_IF_NULL(arg_handler_fg);
  arg_handler_fg->set_manager(fg->manager());
  auto res =
    fg->NewCNodeInOrder({NewValueNode(arg_handler_fg), NewValueNode(op_name), NewValueNode(op_arg.arg_name_), node});
  res->set_debug_info(node->debug_info());
  return res;
}

CNodePtr CheckAndConvertPrimitiveArgs(const PrimitivePtr &prim, const FuncGraphPtr &graph,
                                      const std::pair<std::vector<AnfNodePtr>, std::vector<AnfNodePtr>> &args_pair,
                                      const std::function<AbstractBasePtr(const AnfNodePtr &)> &eval_func,
                                      bool is_preprocessed, const AnfNodePtr &old_cnode = nullptr) {
  auto init_args_list = args_pair.first;
  auto call_args_list = args_pair.second;
  auto prim_name = prim->name();
  auto op_def = mindspore::ops::GetOpDef(prim_name);
  MS_EXCEPTION_IF_NULL(op_def);
  MS_EXCEPTION_IF_NULL(graph);
  // Check args size.
  std::vector<ops::OpInputArg> op_call_args;
  std::vector<ops::OpInputArg> op_init_args;
  auto op_args = op_def->args_;
  for (const auto &op_arg : op_args) {
    if (op_arg.as_init_arg_) {
      (void)op_init_args.emplace_back(op_arg);
    } else {
      (void)op_call_args.emplace_back(op_arg);
    }
  }

  MS_LOG(DEBUG) << "For Primitive[" << prim_name << "], the number of init args is expected to be "
                << op_init_args.size() << ", and the number of call args is expected to be " << op_call_args.size();
  // Generate primitive default args.
  MS_LOG(DEBUG) << "For Primitive[ " << prim_name << "], before processing default args, the number of init args is "
                << init_args_list.size() << " and the number of call args is " << call_args_list.size();
  auto call_nodes = GeneratePrimitiveDefaultArgs(prim_name, call_args_list, op_call_args, eval_func, graph);
  auto init_nodes = GeneratePrimitiveDefaultArgs(prim_name, init_args_list, op_init_args, eval_func, graph);
  MS_LOG(DEBUG) << "For Primitive[ " << prim_name << "], after processing default args, the number of init args is "
                << init_args_list.size() << " and the number of call args is " << call_args_list.size();
  // If it is not preprocessed, signatures and need to be processed.
  if (!is_preprocessed) {
    // Process signatures.
    MS_LOG(DEBUG) << "Process signatures for Primitive[" << prim_name << "].";
    AbstractBasePtrList call_abs_list;
    (void)std::transform(call_nodes.cbegin(), call_nodes.cend(), std::back_inserter(call_abs_list), eval_func);
    call_nodes = prim::GetNewInputsBySignatures(graph, prim_name, prim, call_abs_list, call_nodes, old_cnode);
    // Process arg_handler.
    for (size_t i = 0; i < op_init_args.size(); ++i) {
      auto abs_node = eval_func(init_nodes[i]);
      if (!prim->HasAttr("Converted")) {
        init_nodes[i] = GetNodeAfterArgHandler(init_nodes[i], prim_name, op_init_args[i], abs_node, graph);
      }
    }
  }
  for (size_t i = 0; i < op_call_args.size(); ++i) {
    auto abs_node = eval_func(call_nodes[i]);
    if (!prim->HasAttr("Converted")) {
      call_nodes[i] = GetNodeAfterArgHandler(call_nodes[i], prim_name, op_call_args[i], abs_node, graph);
    }
  }

  // Check args type and do type conversion.
  AbstractBasePtrList call_abs_list;
  AbstractBasePtrList init_abs_list;
  (void)std::transform(call_nodes.cbegin(), call_nodes.cend(), std::back_inserter(call_abs_list), eval_func);
  (void)std::transform(init_nodes.cbegin(), init_nodes.cend(), std::back_inserter(init_abs_list), eval_func);
  MS_LOG(DEBUG) << "For Primitive[" << prim_name << "], the number of init args is " << init_nodes.size()
                << " and the number of call args is " << call_nodes.size();
  if (!ValidateAndConvertArgsType(prim_name, op_call_args, call_abs_list, graph, &call_nodes) ||
      !ValidateAndConvertArgsType(prim_name, op_init_args, init_abs_list, graph, &init_nodes)) {
    std::vector<std::string> op_type_list;
    (void)std::transform(call_abs_list.cbegin(), call_abs_list.cend(), std::back_inserter(op_type_list),
                         [](const AbstractBasePtr &op_abs) { return prim::BuildArgsTypeString(op_abs->BuildType()); });
    (void)std::transform(init_abs_list.cbegin(), init_abs_list.cend(), std::back_inserter(op_type_list),
                         [](const AbstractBasePtr &op_abs) { return prim::BuildArgsTypeString(op_abs->BuildType()); });
    MS_EXCEPTION(TypeError) << prim::BuildOpErrorMsg(op_def, op_type_list);
  }

  // Create New node.
  AnfNodePtrList input_nodes{NewValueNode(prim)};
  (void)std::copy(call_nodes.cbegin(), call_nodes.cend(), std::back_inserter(input_nodes));
  (void)std::copy(init_nodes.cbegin(), init_nodes.cend(), std::back_inserter(input_nodes));
  auto new_cnode = graph->NewCNodeInOrder(input_nodes);
  return new_cnode;
}
}  // namespace

AnfNodePtr CheckAndConvertPrimitiveArgs(const PrimitivePtr &prim,
                                        const std::pair<std::vector<AnfNodePtr>, std::vector<AnfNodePtr>> &args_pair,
                                        const AnalysisEnginePtr &engine, const AnfNodeConfigPtr &out_conf,
                                        bool is_preprocessed) {
  auto node = out_conf->node();
  MS_EXCEPTION_IF_NULL(node);
  auto graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(graph);

  auto eval_func = [&engine, &out_conf](const AnfNodePtr &node) {
    AnfNodeConfigPtr config = engine->MakeConfig(node, out_conf->context(), out_conf->func_graph());
    MS_EXCEPTION_IF_NULL(config);
    const auto &eval_result = config->ObtainEvalResult();
    MS_EXCEPTION_IF_NULL(eval_result);
    return eval_result->abstract();
  };

  auto new_cnode = CheckAndConvertPrimitiveArgs(prim, graph, args_pair, eval_func, is_preprocessed, node);
  MS_LOG(DEBUG) << "Convert primitive args: " << prim->name() << ". node: " << node->DebugString()
                << ", new_node: " << new_cnode->DebugString();
  new_cnode->set_debug_info(node->debug_info());
  return new_cnode;
}

CNodePtr GeneratePrimitiveCNode(const PrimitivePtr &primitive, const ops::OpDef *op_def, const FuncGraphPtr &graph,
                                const AnfNodePtrList &init_args_nodes, const AnfNodePtrList &call_args_nodes,
                                const std::function<AbstractBasePtr(const AnfNodePtr &)> &eval_func) {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(op_def);

  auto args_pair = std::make_pair(init_args_nodes, call_args_nodes);

  // Follow the implementations in PrimitiveArgsToInputsEvaluator, convert to base Primitive, and is_preprocessed=true
  auto new_prim = std::make_shared<Primitive>(*primitive);
  auto new_cnode = CheckAndConvertPrimitiveArgs(new_prim, graph, args_pair, eval_func, true);

  MS_LOG(INFO) << "Convert primitive args: " << primitive->name() << ", new node: " << new_cnode->DebugString();
  return new_cnode;
}

std::shared_ptr<Functional> BuildMethodFunctional(const std::string &name) {
  auto functional = std::make_shared<Functional>(name);
  functional->set_is_method(true);
  return functional;
}

namespace {
bool IsSubtypeTuple(const AbstractBasePtr x, const TypePtr model) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(model);
  auto x_tuple = dyn_cast_ptr<AbstractTuple>(x);
  auto model_tuple = dyn_cast_ptr<Tuple>(model);

  if (x_tuple == nullptr || model_tuple == nullptr) {
    return false;
  }

  if (model->IsGeneric()) {
    return true;
  }

  if (x_tuple->size() != model_tuple->size()) {
    return false;
  }

  for (size_t i = 0; i < x_tuple->size(); i++) {
    bool is_subtype = IsSubtype((*x_tuple)[i], (*model_tuple)[i]);
    if (!is_subtype) {
      return false;
    }
  }
  return true;
}

bool IsSubtypeArray(const AbstractBasePtr x, const TypePtr model) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(model);
  auto x_tensor = dyn_cast_ptr<AbstractTensor>(x);
  auto model_tensor = dyn_cast_ptr<TensorType>(model);

  if (x_tensor == nullptr || model_tensor == nullptr) {
    return false;
  }

  if (model->IsGeneric()) {
    return true;
  }

  return IsSubtype(x_tensor->element(), model_tensor->element());
}

bool IsSubtypeList(const AbstractBasePtr x, const TypePtr model) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(model);
  auto x_list = dyn_cast_ptr<AbstractList>(x);
  auto model_list = dyn_cast_ptr<List>(model);

  if (x_list == nullptr || model_list == nullptr) {
    return false;
  }

  if (model->IsGeneric()) {
    return true;
  }

  if (x_list->size() != model_list->size()) {
    return false;
  }

  bool is_subtype = true;
  for (size_t i = 0; i < x_list->size(); i++) {
    is_subtype = IsSubtype((*x_list)[i], (*model_list)[i]);
    if (!is_subtype) {
      return false;
    }
  }
  return is_subtype;
}

inline bool IsSubtypeScalar(const AbstractBasePtr x, const TypePtr model) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(model);
  if (dyn_cast_ptr<AbstractScalar>(x) == nullptr) {
    return false;
  }
  auto &x_type = x->GetTypeTrack();
  return IsSubType(x_type, model);
}
}  // namespace

bool IsSubtype(const AbstractBasePtr x, const TypePtr model) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(model);
  TypeId model_typeid = model->type_id();
  switch (model_typeid) {
    case kMetaTypeObject:
      return true;
    case kObjectTypeTuple:
      return IsSubtypeTuple(x, model);
    case kObjectTypeTensorType:
      return IsSubtypeArray(x, model);
    case kObjectTypeList:
      return IsSubtypeList(x, model);
    default:
      if (IsSubType(model, std::make_shared<Number>())) {
        return IsSubtypeScalar(x, model);
      }
      MS_LOG(EXCEPTION) << "Invalid model type: " << model->ToString() << ".";
  }
}

template <typename T>
bool HasAbstractType(const AbstractBasePtr &abs) {
  if (abs->isa<AbstractSequence>()) {
    auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    return std::any_of(abs_seq->elements().cbegin(), abs_seq->elements().cend(), HasAbstractType<T>);
  }
  return abs->IsSameTypeId(T::kTypeId);
}

}  // namespace abstract
}  // namespace mindspore
