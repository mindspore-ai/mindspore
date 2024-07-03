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

#include "pipeline/jit/pi/graph_build/func_graph_builder.h"
#include <algorithm>
#include <utility>
#include <set>
#include <queue>
#include "frontend/operator/composite/do_signature.h"
#include "pipeline/jit/ps/static_analysis/static_analysis.h"
#include "pipeline/jit/ps/action.h"
#include "pipeline/jit/ps/parse/parse_base.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "pipeline/jit/pi/pi_jit_config.h"
#include "pipeline/jit/pi/graph_guard/infer.h"
#include "ops/arithmetic_ops.h"
#include "ops/structure_ops.h"
#include "include/common/utils/convert_utils_py.h"
#include "ir/tensor.h"
#include "ir/anf.h"
#include "frontend/operator/composite/unpack_call.h"

namespace mindspore {
namespace {
constexpr auto kPiJitPyObjKey = "pi_jit_py_obj";
constexpr auto kGradFuncPyObject = "grad_func_py_obj";
constexpr auto kGradNetInputs = "grad_net_inputs";
constexpr auto kTensorModule = "mindspore.common";
constexpr auto kAdapterFlag = "adapter_flag";
constexpr auto kInnerOpsModule = "mindspore.ops.operations._inner_ops";

bool ShouldFallBackInRuntime(const PrimitivePtr &prim) {
  static HashSet<std::string> prims_should_fallback_in_runtime = {kListInplaceExtendOpName,
                                                                  kListInplaceInsertOpName,
                                                                  kListInplacePopOpName,
                                                                  kListInplaceReverseOpName,
                                                                  kListInplaceClearOpName,
                                                                  kDictInplaceSetItemOpName,
                                                                  kRaiseOpName,
                                                                  kJoinedStrOpName,
                                                                  kFormatOpName};
  return prims_should_fallback_in_runtime.find(prim->name()) != prims_should_fallback_in_runtime.end();
}

bool IsValidScalar(const AbstractBasePtr &abs) {
  auto build_type = abs->BuildType();
  return build_type->isa<String>() || build_type->isa<Number>();
}

bool Mutable(const py::object &obj, const ValuePtr &value = nullptr) {
  // If a tensor has been set const arg, it should not be mutable.
  if (value != nullptr && value->isa<tensor::MetaTensor>()) {
    constexpr char const_arg_attr[] = "const_arg";
    if (py::hasattr(obj, const_arg_attr) && py::cast<bool>(py::getattr(obj, const_arg_attr))) {
      return false;
    }
  }
  constexpr char mutable_attr[] = "__ms_mutable__";
  return py::hasattr(obj, mutable_attr) && py::cast<bool>(py::getattr(obj, mutable_attr));
}

bool IsParameter(const py::object &obj) {
  return py::hasattr(obj, "__parameter__") && py::isinstance<tensor::MetaTensor>(obj);
}

bool TensorArgMutable(const py::object &obj, const ValuePtr &value) {
  if (!value->isa<tensor::MetaTensor>()) {
    return false;
  }
  constexpr char const_arg_attr[] = "const_arg";
  return !py::hasattr(obj, const_arg_attr) || !py::cast<bool>(py::getattr(obj, const_arg_attr));
}

bool NeedBroaden(const py::object &obj, const ValuePtr &value) {
  return TensorArgMutable(obj, value) || Mutable(obj, value) || value->isa<tensor::MetaSparseTensor>();
}

TypeId GetTypeIdFromClassName(const std::string &class_name) {
  static HashMap<std::string, TypeId> class_name_to_type_ids = {
    {"Tensor", kObjectTypeTensorType},  {"list", kObjectTypeList},
    {"tuple", kObjectTypeTuple},        {"int", kNumberTypeInt},
    {"float", kNumberTypeFloat},        {"CellList", kObjectTypeList},
    {"CellDict", kObjectTypeDictionary}};
  auto iter = class_name_to_type_ids.find(class_name);
  if (iter == class_name_to_type_ids.end()) {
    return kTypeUnknown;
  }
  return iter->second;
}

bool FunctionShouldBeParseInAst(const py::object &obj) {
  static mindspore::HashSet<std::string> func_names{"cast_to_adapter_tensor", "cast_to_ms_tensor"};
  if (!py::hasattr(obj, "__name__")) {
    return false;
  }
  return func_names.find(py::cast<std::string>(obj.attr("__name__"))) != func_names.end();
}
}  // namespace

ValuePtr FuncGraphBuilder::ConvertPyObjToValue(const py::object &obj) {
  if (obj.ptr() == nullptr) {
    return nullptr;
  }
  ValuePtr ret = nullptr;
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    if (!parse::ConvertData(obj, &ret)) {
      return nullptr;
    }
  } catch (const std::exception &e) {
    MS_LOG(DEBUG) << "Failed to convert python object << " << py::str(obj) << " to value. The exception:\n" << e.what();
    return nullptr;
  }
  return ret;
}

AnfNodePtr FuncGraphBuilder::ConvertParameterTupleToNode(const py::object &input_obj) {
  constexpr auto parameter_tuple_attr = "__parameter_tuple__";
  if (input_obj.ptr() == nullptr || !py::hasattr(input_obj, parameter_tuple_attr)) {
    return nullptr;
  }
  auto tuple_obj = input_obj.cast<py::tuple>();
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim::kPrimMakeTuple)};
  std::vector<AbstractBasePtr> inputs_abs;
  for (const auto &obj : tuple_obj) {
    if (!IsParameter(py::cast<py::object>(obj))) {
      MS_LOG(INFO) << "Encounter non parameter object in parameter tuple object: " << py::str(obj);
      return nullptr;
    }
    auto cur_node = parse::ResolveParameterObj(graph_, py::cast<py::object>(obj));
    if (cur_node == nullptr) {
      return nullptr;
    }
    auto cur_abs = cur_node->abstract();
    if (cur_abs == nullptr) {
      return nullptr;
    }
    inputs.push_back(cur_node);
    inputs_abs.push_back(cur_abs);
  }
  auto ret = graph_->NewCNodeInOrder(inputs);
  auto ret_abs = std::make_shared<abstract::AbstractTuple>(inputs_abs);
  ret->set_abstract(ret_abs);
  MS_LOG(INFO) << "Convert parameter tuple to node: " << ret->DebugString()
               << " with abstract: " << ret_abs->ToString();
  return ret;
}

AnfNodePtr FuncGraphBuilder::ConvertObjToNode(const py::object &input_obj) {
  if (IsParameter(input_obj)) {
    // Add the fv parameter and set its abstract.
    return parse::ResolveParameterObj(graph_, input_obj);
  }
  auto parameter_tuple_object = ConvertParameterTupleToNode(input_obj);
  if (parameter_tuple_object != nullptr) {
    return parameter_tuple_object;
  }
  auto val = ConvertPyObjToValue(input_obj);
  if (val == nullptr) {
    MS_LOG(INFO) << "The input object " << py::str(input_obj) << " convert to value failed.";
    return nullptr;
  }
  // Constant value input scene, the object should be converted to value node.
  auto node = NewValueNode(val);
  node->set_abstract(val->ToAbstract());
  return node;
}

AbstractBasePtr FuncGraphBuilder::EvalValue(const ValuePtr &value, const AbstractBasePtrList &inputs_abs_list) {
  if (value == nullptr) {
    return nullptr;
  }
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    if (value->isa<Primitive>()) {
      auto prim = value->cast<PrimitivePtr>();
      auto eval_res = abstract::EvalOnePrim(prim, inputs_abs_list);
      if (eval_res != nullptr) {
        return eval_res->abstract();
      }
    } else if (value->ToAbstract()->isa<abstract::AbstractFunction>()) {
      auto analyze_res = pipeline::AbstractAnalyze(value, inputs_abs_list);
      if (analyze_res.eval_result != nullptr) {
        return analyze_res.eval_result->abstract();
      }
    }
    return nullptr;
  } catch (const std::exception &e) {
    MS_LOG(INFO) << "Failed to EvalValue for value: " << value->ToString();
    return nullptr;
  }
}

bool FuncGraphBuilder::CheckCallable(const ValuePtr &value, const AbstractBasePtr &abs) {
  if (value == nullptr || abs == nullptr || abs->isa<abstract::AbstractAny>()) {
    return false;
  }
  if (value->isa<Primitive>() && ShouldFallBackInRuntime(value->cast<PrimitivePtr>())) {
    return false;
  }
  return true;
}

bool FuncGraphBuilder::CheckGraphOutput(const AbstractBasePtr &abs) {
  if (abs == nullptr) {
    return false;
  }
  if (abs->isa<abstract::AbstractSequence>()) {
    const auto elements = abs->cast<abstract::AbstractSequencePtr>()->elements();
    return std::all_of(elements.begin(), elements.end(),
                       [](const AbstractBasePtr &elem) { return CheckGraphOutput(elem); });
  }
  if (abs->isa<abstract::AbstractScalar>()) {
    return IsValidScalar(abs);
  }
  return abs->isa<abstract::AbstractTensor>() || abs->isa<abstract::AbstractRowTensor>() ||
         abs->isa<abstract::AbstractMapTensor>();
}

AbstractWrapperPtr FuncGraphBuilder::AddLocalVariable(const py::object &obj) {
  if (obj.ptr() == nullptr) {
    MS_LOG(INFO) << "Failed to add local variable, py object is null";
    return nullptr;
  }

  auto node = ConvertObjToNode(obj);
  if (node == nullptr) {
    MS_LOG(INFO) << "Failed to add local variable, convert python object to anf node failed";
    return nullptr;
  }
  auto abstract_wrapper = std::make_shared<AbstractWrapper>(node->abstract());

  (void)key_to_node_.emplace(abstract_wrapper, node);
  return abstract_wrapper;
}

AnfNodePtr FuncGraphBuilder::ReadLocalVariable(const AbstractWrapperPtr &abstract_wrapper) {
  auto iter = key_to_node_.find(abstract_wrapper);
  if (iter == key_to_node_.end()) {
    return nullptr;
  }
  return iter->second;
}

AnfNodePtr FuncGraphBuilder::GetNodeByWrapper(const AbstractWrapperPtr &abstract_wrapper) {
  // Search the predecessors of the current builder for the local parameter with BFS.
  if (abstract_wrapper == nullptr || abstract_wrapper->abstract() == nullptr) {
    return nullptr;
  }
  mindspore::HashSet<FuncGraphBuilder *> visited_builders;
  std::queue<FuncGraphBuilder *> builder_queue;
  builder_queue.push(this);
  while (!builder_queue.empty()) {
    const auto cur_builder = builder_queue.front();
    MS_EXCEPTION_IF_NULL(cur_builder);
    builder_queue.pop();
    (void)visited_builders.insert(cur_builder);
    auto node = cur_builder->ReadLocalVariable(abstract_wrapper);
    if (node != nullptr) {
      MS_LOG(INFO) << "Found node: " << node->DebugString()
                   << " for abstract wrapper: " << abstract_wrapper->ToString();
      return node;
    }
    for (const auto &cur_pred_builder : cur_builder->prev_builders()) {
      if (visited_builders.count(cur_pred_builder) == 0) {
        builder_queue.push(cur_pred_builder);
      }
    }
  }
  // Build ValueNode for constant abstract.
  // Need to handle tuple/list/dict with FuncGraphAbstractClosure scene later.
  auto abstract = abstract_wrapper->abstract();
  if (abstract->isa<abstract::FuncGraphAbstractClosure>()) {
    auto abs_func = abstract->cast<abstract::FuncGraphAbstractClosurePtr>();
    auto fg = abs_func->func_graph();
    return NewValueNode(fg);
  }
  auto value = abstract->BuildValue();
  if (value != kValueAny) {
    auto ret = NewValueNode(value);
    ret->set_abstract(abstract);
    return ret;
  }
  return nullptr;
}

AbstractWrapperPtr FuncGraphBuilder::AddTopGraphArgInput(const py::object &object) {
  if (object.ptr() == nullptr) {
    MS_LOG(INFO) << "Get top graph arg input failed.";
    return nullptr;
  }
  auto value = ConvertPyObjToValue(object);
  if (value == nullptr) {
    return nullptr;
  }
  bool broaden = NeedBroaden(object, value);
  AbstractBasePtr abs = abstract::ToAbstract(value, nullptr, nullptr);
  if (broaden) {
    abs = AbstractBroaden(abs);
  }
  if (abs == nullptr) {
    MS_LOG(INFO) << "Failed to add input for python object: " << std::string(py::str(object)) << "  " << object.ptr();
    return nullptr;
  }
  auto para = graph_->add_parameter();
  para->set_abstract(abs);
  para->set_is_top_graph_param(true);
  para->set_user_data(kPiJitPyObjKey, std::make_shared<py::object>(object));
  AbstractWrapperPtr abstract_wrapper = std::make_shared<AbstractWrapper>(para->abstract());
  (void)key_to_node_.emplace(abstract_wrapper, para);
  MS_LOG(INFO) << "Add top arg input success, python object: " << py::str(object) << ", node: " << para->DebugString()
               << ", abstract: " << abs->ToString();
  return abstract_wrapper;
}

AbstractWrapperPtr FuncGraphBuilder::AddTopGraphVargsInputs(const py::object &vargs) {
  if (vargs.ptr() == nullptr) {
    MS_LOG(INFO) << "Top graph vargs is nullptr.";
    return nullptr;
  }
  auto vargs_tuple = vargs.cast<py::tuple>();
  if (vargs_tuple.ptr() == nullptr) {
    MS_LOG(INFO) << "Vargs object should be tuple but got: " << py::str(vargs) << ", add top graph vargs failed.";
    return nullptr;
  }
  auto value = ConvertPyObjToValue(vargs);
  if (value == nullptr || !value->isa<ValueTuple>()) {
    MS_LOG(INFO) << "Convert vargs to value failed, vargs: " << py::str(vargs);
    return nullptr;
  }
  auto value_tuple = value->cast<ValueTuplePtr>();
  const auto &elements = value_tuple->value();
  if (elements.size() != vargs_tuple.size()) {
    MS_LOG(INFO) << "For top graph vargs, converted value element size is " << elements.size()
                 << ", python tuple element size is " << vargs_tuple.size() << ". Size not matched.";
    return nullptr;
  }
  std::vector<AbstractBasePtr> new_elements;
  auto para = graph_->add_parameter();
  for (size_t i = 0; i < elements.size(); ++i) {
    auto cur_obj = vargs_tuple[i].cast<py::object>();
    auto cur_val = elements[i];
    bool broaden = NeedBroaden(cur_obj, cur_val);
    auto cur_abs = abstract::ToAbstract(cur_val, nullptr, nullptr);
    if (broaden) {
      cur_abs = AbstractBroaden(cur_abs);
    }
    if (cur_abs == nullptr) {
      MS_LOG(INFO) << "Fail to convert args element " << cur_val->ToString();
      return nullptr;
    }
    new_elements.push_back(cur_abs);
  }
  auto new_vargs_abs = std::make_shared<abstract::AbstractTuple>(new_elements);
  para->set_abstract(new_vargs_abs);
  para->set_is_top_graph_param(true);
  para->set_user_data(kPiJitPyObjKey, std::make_shared<py::object>(vargs));
  AbstractWrapperPtr abstract_wrapper = std::make_shared<AbstractWrapper>(para->abstract());
  (void)key_to_node_.emplace(abstract_wrapper, para);
  MS_LOG(INFO) << "Add top vargs input success, python object: " << py::str(vargs) << ", node: " << para->DebugString()
               << ", abstract: " << new_vargs_abs->ToString();
  return abstract_wrapper;
}

AbstractWrapperPtr FuncGraphBuilder::AddTopGraphKwargsInputs(const py::object &kwargs) {
  if (kwargs.ptr() == nullptr) {
    MS_LOG(INFO) << "Top graph kwargs input is nullptr.";
    return nullptr;
  }
  auto kwargs_dict = kwargs.cast<py::dict>();
  if (kwargs_dict.ptr() == nullptr) {
    MS_LOG(INFO) << "Kwargs object should be tuple but got: " << py::str(kwargs) << ", add top graph kwargs failed.";
    return nullptr;
  }
  auto value = ConvertPyObjToValue(kwargs);
  if (value == nullptr || !value->isa<ValueDictionary>()) {
    MS_LOG(INFO) << "Convert kwargs to value failed, kwargs: " << py::str(kwargs);
    return nullptr;
  }
  auto value_dict = value->cast<ValueDictionaryPtr>();
  const auto &elements = value_dict->value();
  if (elements.size() != kwargs_dict.size()) {
    MS_LOG(INFO) << "Kwargs dict size is " << kwargs_dict.size() << " and corresponding value dict size is "
                 << elements.size() << ". Size not matched.";
  }
  auto para = graph_->add_parameter();
  std::vector<abstract::AbstractElementPair> new_key_values;
  for (size_t i = 0; i < elements.size(); ++i) {
    auto cur_key_val = elements[i].first;
    auto cur_val = elements[i].second;
    auto cur_key_obj = ValueToPyData(cur_key_val);
    if (!kwargs_dict.contains(cur_key_obj)) {
      return nullptr;
    }
    auto cur_val_obj = kwargs_dict[cur_key_obj];
    auto cur_value_abs = abstract::ToAbstract(cur_val, nullptr, nullptr);
    bool broaden = NeedBroaden(cur_val_obj, cur_val);
    if (broaden) {
      cur_value_abs = AbstractBroaden(cur_value_abs);
    }
    if (cur_value_abs == nullptr) {
      MS_LOG(INFO) << "Fail to convert kwargs value element " << cur_val->ToString();
      return nullptr;
    }
    auto cur_key_abs = abstract::ToAbstract(cur_key_val, nullptr, nullptr);
    new_key_values.push_back(abstract::AbstractElementPair{cur_key_abs, cur_value_abs});
  }
  auto new_kwargs_abs = std::make_shared<abstract::AbstractDictionary>(new_key_values);
  para->set_abstract(new_kwargs_abs);
  para->set_is_top_graph_param(true);
  para->set_user_data(kPiJitPyObjKey, std::make_shared<py::object>(kwargs));
  AbstractWrapperPtr abstract_wrapper = std::make_shared<AbstractWrapper>(para->abstract());
  (void)key_to_node_.emplace(abstract_wrapper, para);
  MS_LOG(INFO) << "Add top kwargs input success, python object: " << py::str(kwargs)
               << ", node: " << para->DebugString() << ", abstract: " << new_kwargs_abs->ToString();
  return abstract_wrapper;
}

AbstractWrapperPtr FuncGraphBuilder::AddSubGraphInput(const AbstractWrapperPtr abstract_wrapper) {
  MS_LOG(INFO) << "Try add sub graph parameter for abstract wrapper: " << abstract_wrapper->ToString();
  if (abstract_wrapper == nullptr) {
    MS_LOG(INFO) << "Abstract wrapper for subgraph input is nullptr.";
    return nullptr;
  }
  auto node = GetNodeByWrapper(abstract_wrapper);
  AbstractBasePtr para_abs = node->abstract();
  if (para_abs == nullptr) {
    MS_LOG(INFO) << "Failed to add input for abstract wrapper: " << abstract_wrapper->ToString();
    return nullptr;
  }
  auto para = graph_->add_parameter();
  para->set_abstract(para_abs);
  para->set_is_top_graph_param(false);
  AbstractWrapperPtr ret_abstract_wrapper =
    abstract_wrapper == nullptr ? std::make_shared<AbstractWrapper>(para->abstract()) : abstract_wrapper;
  (void)key_to_node_.emplace(ret_abstract_wrapper, para);
  MS_LOG(INFO) << "Add input success for abstract wrapper: " << abstract_wrapper->ToString()
               << ", result abstract wrapper: " << ret_abstract_wrapper->ToString();
  return ret_abstract_wrapper;
}

AbstractWrapperPtr FuncGraphBuilder::AddNode(const py::object &callable_obj,
                                             const std::vector<AbstractWrapperPtr> &inputs_abstract_wrapper) {
  if (!CheckCallable(callable_obj)) {
    MS_LOG(INFO) << "The python obj " << py::str(callable_obj) << " is not callable.";
    return nullptr;
  }
  auto callable_value = ConvertPyObjToValue(callable_obj);
  if (callable_value == nullptr) {
    MS_LOG(INFO) << "Convert python object " << py::str(callable_obj) << " to value failed.";
    return nullptr;
  }

  const std::string &callable_str = callable_value->ToString();
  const std::string grad_prefix = "MetaFuncGraph-grad";
  if (callable_str.substr(0, grad_prefix.size()) == grad_prefix) {
    MS_LOG(INFO) << "Grad scene callable: " << callable_str;
    return BuildGradNetNode(callable_value, callable_obj, inputs_abstract_wrapper);
  }

  if (FunctionShouldBeParseInAst(callable_obj)) {
    return TryToAddNode(callable_value, inputs_abstract_wrapper);
  }
  return AddNode(callable_value, inputs_abstract_wrapper);
}

AbstractWrapperPtr FuncGraphBuilder::AddAttrPythonObject(const py::object &object) {
  if (object.ptr() == nullptr) {
    MS_LOG(INFO) << "Convert python object with empty object, convert failed.";
    return nullptr;
  }
  // Attribute object is constant or Parameter, do not need to check constant.
  auto node = ConvertObjToNode(object);
  if (node == nullptr || node->abstract() == nullptr) {
    MS_LOG(INFO) << "Convert python object " << py::str(object) << " to anf node failed.";
    return nullptr;
  }
  auto abstract_wrapper = std::make_shared<AbstractWrapper>(node->abstract());
  (void)key_to_node_.emplace(abstract_wrapper, node);
  return abstract_wrapper;
}

bool FuncGraphBuilder::GetInputNodesAndAbstracts(const ValuePtr &callable_value,
                                                 const std::vector<AbstractWrapperPtr> &inputs_abstract_wrapper,
                                                 std::vector<AnfNodePtr> *input_node_list,
                                                 std::vector<AbstractBasePtr> *input_abs_list) {
  input_node_list->reserve(inputs_abstract_wrapper.size() + 1);
  input_abs_list->reserve(inputs_abstract_wrapper.size());

  (void)input_node_list->emplace_back(NewValueNode(callable_value));
  for (const auto &input_wrapper : inputs_abstract_wrapper) {
    if (input_wrapper == nullptr) {
      MS_LOG(INFO) << "The input python object of " << callable_value->ToString() << ", is NULL";
      return false;
    }
    auto node = GetNodeByWrapper(input_wrapper);
    if (node == nullptr) {
      return false;
    }
    (void)input_node_list->emplace_back(node);
    (void)input_abs_list->emplace_back(node->abstract());
  }
  return true;
}

CNodePtr FuncGraphBuilder::DoPrimitiveInferAndCheck(const PrimitivePtr &primitive,
                                                    const AnfNodePtrList &input_node_list,
                                                    const AbstractBasePtrList &args_abs_list) {
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    const CNodePtr &new_node = AddPrimitiveCNode(primitive, input_node_list, args_abs_list);
    if (new_node == nullptr) {
      MS_LOG(INFO) << "Failed to add CNode for Primitive: " << primitive->name();
      return nullptr;
    }

    const AbstractBasePtr &abs = GetAbstractOf(new_node);

    if (!CheckCallable(primitive, abs)) {
      MS_LOG(INFO) << "Check callable failed for Primitive: " << primitive->name();
      return nullptr;
    }
    new_node->set_abstract(abs);
    return new_node;
  } catch (const std::exception &e) {
    MS_LOG(INFO) << "Failed to infer Primitive: " << primitive->name() << ". The exception:\n" << e.what();
    return nullptr;
  }
}

CNodePtr FuncGraphBuilder::AddPrimitiveCNode(const PrimitivePtr &primitive, const AnfNodePtrList &input_node_list,
                                             const AbstractBasePtrList &args_abs_list) {
  auto op_def = mindspore::ops::GetOpDef(primitive->name());

  if (op_def == nullptr) {
    if (primitive->has_signature()) {
      // Follow the implementations in DoSignatureEvaluator
      AnfNodePtrList args_node_list(input_node_list.cbegin() + 1, input_node_list.cend());
      AnfNodePtrList new_node_list =
        prim::GetNewInputsBySignatures(graph_, primitive->ToString(), primitive, args_abs_list, args_node_list);

      new_node_list.insert(new_node_list.begin(), input_node_list[0]);
      return graph_->NewCNodeInOrder(new_node_list);
    }
  } else if (primitive->isa<PrimitivePy>()) {
    // Follow the implementations in PrimitiveArgsToInputsEvaluator and DoTransPrimitiveFunctionEvaluator
    auto arg_signatures = op_def->signatures_;
    primitive->set_signatures(arg_signatures);
    primitive->set_has_signature(!arg_signatures.empty());

    const AnfNodePtrList &init_args = abstract::GetPrimitiveInitArgs(primitive->cast<PrimitivePyPtr>(), op_def);

    AnfNodePtrList call_args(input_node_list.cbegin() + 1, input_node_list.cend());
    AbstractBasePtrList call_abs_list;
    (void)std::transform(call_args.cbegin(), call_args.cend(), std::back_inserter(call_abs_list),
                         [](const AnfNodePtr &node) { return FuncGraphBuilder::GetAbstractOf(node); });
    const AnfNodePtrList &new_call_args =
      prim::GetNewInputsBySignatures(graph_, primitive->name(), primitive, call_abs_list, call_args);

    return abstract::GeneratePrimitiveCNode(
      primitive, op_def, graph_, init_args, new_call_args,
      [](const AnfNodePtr &node) { return FuncGraphBuilder::GetAbstractOf(node); });
  }
  MS_LOG(DEBUG) << "Primitive " << primitive->name() << " no need to process signatures and OpDef";
  return graph_->NewCNodeInOrder(input_node_list);
}

AbstractBasePtr FuncGraphBuilder::GetAbstractOf(const AnfNodePtr &node) {
  if (node == nullptr) {
    return nullptr;
  }
  if (node->abstract() != nullptr) {
    return node->abstract();
  }
  if (node->isa<ValueNode>()) {
    return node->cast<ValueNodePtr>()->value()->ToAbstract();
  } else if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode->empty() || !cnode->input(0)->isa<ValueNode>()) {
      return nullptr;
    }
    ValuePtr value = cnode->input(0)->cast<ValueNodePtr>()->value();
    std::vector<AbstractBasePtr> abs_list;
    std::transform(cnode->inputs().begin() + 1, cnode->inputs().end(), std::back_inserter(abs_list),
                   [](const AnfNodePtr &node) {
                     if (node->abstract() == nullptr) {
                       node->set_abstract(FuncGraphBuilder::GetAbstractOf(node));
                     }
                     return node->abstract();
                   });
    return EvalValue(value, abs_list);
  }
  MS_LOG(INFO) << "Unsupported Node type for GetAbstractOf() method, node: " << node->DebugString();
  return nullptr;
}

AbstractBasePtr FuncGraphBuilder::DoInferAndCheck(const ValuePtr &callable_value,
                                                  const vector<AbstractBasePtr> &input_abs_list) {
  auto abs = EvalValue(callable_value, input_abs_list);
  if (abs == nullptr) {
    MS_LOG(DEBUG) << "Eval failed for value: " << callable_value->ToString();
    return nullptr;
  }
  if (!CheckCallable(callable_value, abs)) {
    MS_LOG(DEBUG) << "Check callable failed for value: " << callable_value->ToString() << ", abs: " << abs->ToString();
    return nullptr;
  }
  return abs;
}

AbstractWrapperPtr FuncGraphBuilder::BuildGradNetNode(const ValuePtr &callable_value, const py::object &callable_obj,
                                                      const std::vector<AbstractWrapperPtr> &inputs_abstract_wrapper) {
  const std::string grad_prefix = "MetaFuncGraph-grad";
  const std::string fake_node_key_prefix = "FakeNodeKey";
  std::vector<AnfNodePtr> input_node_list;
  std::vector<AbstractBasePtr> input_abs_list;
  if (!GetInputNodesAndAbstracts(callable_value, inputs_abstract_wrapper, &input_node_list, &input_abs_list)) {
    return nullptr;
  }
  auto fake_node = graph_->NewCNode(input_node_list);
  constexpr auto forward_fg_index = 1;
  auto forward_fg = GetValueNode<FuncGraphPtr>(input_node_list[forward_fg_index]);
  MS_EXCEPTION_IF_NULL(forward_fg);
  auto origin_forward_fg_output = forward_fg->output();
  std::stringstream ss;
  ss << fake_node.get();
  auto output_py_obj = py::str(fake_node_key_prefix + " " + grad_prefix + " " + ss.str());

  auto abs = abstract::ToAbstract(MakeValue(ConvertPyObjToValue(output_py_obj)));
  abs->set_user_data(kGradNetInputs, std::make_shared<std::vector<AbstractWrapperPtr>>(inputs_abstract_wrapper));
  abs->set_user_data(kGradFuncPyObject, std::make_shared<py::object>(callable_obj));
  fake_node->set_abstract(abs);
  auto cur_forward_fg_output = forward_fg->output();
  if (origin_forward_fg_output != cur_forward_fg_output) {
    // has_aux for GradOperation will change the output of forward fg.
    forward_fg->set_output(origin_forward_fg_output);
  }
  auto abstract_wrapper = std::make_shared<AbstractWrapper>(fake_node->abstract());
  (void)key_to_node_.emplace(abstract_wrapper, fake_node);
  MS_LOG(INFO) << "Build GradOperation Net fake node: " << fake_node->DebugString();
  return abstract_wrapper;
}

AbstractWrapperPtr FuncGraphBuilder::BuildGradNode(const AbstractWrapperPtr &key,
                                                   const std::vector<AbstractWrapperPtr> &inputs, bool need_unpack) {
  AbstractWrapperPtr ret;
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    ret = HandleGrad(key, inputs, need_unpack);
  } catch (const std::exception &e) {
    MS_LOG(INFO) << "Failed to build grad node with key: " << key << ". The exception:\n" << e.what();
  }
  return ret;
}

// For GradOperation(net, ...)(forward_inputs), two nodes should be evaluated together as a graph.
// Before:
//   fake_node: GradOperation(net, other_inputs)
// After:
//   fg(other_inputs, forward_inputs)
//     grad_net_node:    DoSignature(GradOperation)(net, other_inputs)
//     grad_result_node: grad_net_node(forward_inputs) or unpack_call(grad_net_node, forward_inputs)
//     return grad_result_node
//   final node for evaluated: fg(other_inputs, forward_inputs)
AbstractWrapperPtr FuncGraphBuilder::HandleGrad(const AbstractWrapperPtr &key,
                                                const std::vector<AbstractWrapperPtr> &inputs, bool need_unpack) {
  auto fake_node = ReadLocalVariable(key);
  if (fake_node == nullptr || !fake_node->isa<CNode>()) {
    MS_LOG(INFO) << "Failed to find corresponding fake GradOperation node for key: " << key;
    return nullptr;
  }
  auto fake_node_abstract = fake_node->abstract();
  if (fake_node_abstract == nullptr) {
    MS_LOG(INFO) << "When handling grad, fail to find abstract for fake node: " << fake_node->DebugString();
    return nullptr;
  }
  if (!fake_node_abstract->has_user_data(kGradNetInputs) || !fake_node_abstract->has_user_data(kGradFuncPyObject)) {
    MS_LOG(INFO) << "When handing grad, fail to find corresponding user data for fake node: "
                 << fake_node->DebugString();
    return nullptr;
  }

  auto pre_wrapper = *(fake_node_abstract->user_data<std::vector<AbstractWrapperPtr>>(kGradNetInputs));
  std::vector<AnfNodePtr> fake_node_inputs;
  for (auto e : pre_wrapper) {
    auto cur_node = GetNodeByWrapper(e);
    MS_EXCEPTION_IF_NULL(cur_node);
    fake_node_inputs.push_back(cur_node);
  }

  auto meta_object = *(fake_node_abstract->user_data<py::object>(kGradFuncPyObject));
  auto value = ConvertPyObjToValue(meta_object);
  MS_EXCEPTION_IF_NULL(value);
  auto meta = value->cast<MetaFuncGraphPtr>();
  MS_EXCEPTION_IF_NULL(meta);

  auto forward_fg_node = fake_node_inputs[0];
  MS_EXCEPTION_IF_NULL(forward_fg_node);
  auto forward_fg = GetValueNode<FuncGraphPtr>(forward_fg_node);
  MS_EXCEPTION_IF_NULL(forward_fg);
  auto origin_forward_fg_output = forward_fg->output();
  auto fake_cnode = fake_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(fake_cnode);
  auto meta_node = NewValueNode(std::make_shared<prim::DoSignaturePrimitive>(meta->name(), meta));
  std::vector<AnfNodePtr> grad_net_node_inputs{meta_node, forward_fg_node};
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  for (size_t i = 1; i < fake_node_inputs.size(); ++i) {
    (void)grad_net_node_inputs.emplace_back(fg->add_parameter());
  }
  auto grad_net_node = fg->NewCNodeInOrder(grad_net_node_inputs);
  std::vector<AnfNodePtr> grad_result_node_inputs;
  if (need_unpack) {
    auto unpack_call_op = NewValueNode(std::make_shared<prim::UnpackCall>(parse::NAMED_METAGRAPH_UNPACKCALL));
    grad_result_node_inputs.push_back(unpack_call_op);
  }
  grad_result_node_inputs.push_back(grad_net_node);
  for (size_t i = 0; i < inputs.size(); ++i) {
    (void)grad_result_node_inputs.emplace_back(fg->add_parameter());
  }
  auto grad_result_node = fg->NewCNodeInOrder(grad_result_node_inputs);
  fg->set_output(grad_result_node);
  std::vector<AnfNodePtr> final_node_input = {NewValueNode(fg)};
  std::vector<AbstractBasePtr> final_node_abs;
  for (size_t i = 1; i < fake_node_inputs.size(); ++i) {
    AnfNodePtr cur_input = fake_node_inputs[i];
    MS_EXCEPTION_IF_NULL(cur_input);
    auto cur_input_abs = cur_input->abstract();
    MS_EXCEPTION_IF_NULL(cur_input_abs);
    final_node_input.push_back(cur_input);
    final_node_abs.push_back(cur_input_abs);
  }
  for (auto input_wrapper : inputs) {
    auto node = GetNodeByWrapper(input_wrapper);
    MS_EXCEPTION_IF_NULL(node);
    (void)final_node_input.emplace_back(node);
    (void)final_node_abs.emplace_back(node->abstract());
  }
  auto final_node = graph_->NewCNodeInOrder(final_node_input);
  fg->set_manager(mng_);
  auto analyze_res = pipeline::AbstractAnalyze(fg, final_node_abs);
  MS_EXCEPTION_IF_NULL(analyze_res.eval_result);
  auto final_abs = analyze_res.eval_result->abstract();
  MS_EXCEPTION_IF_NULL(final_abs);
  final_node->set_abstract(final_abs);
  auto cur_forward_fg_output = forward_fg->output();
  if (origin_forward_fg_output != cur_forward_fg_output) {
    // has_aux for GradOperation will change the output of forward fg.
    forward_fg->set_output(origin_forward_fg_output);
  }

  auto abstract_wrapper = std::make_shared<AbstractWrapper>(final_node->abstract());
  (void)key_to_node_.emplace(abstract_wrapper, final_node);
  return abstract_wrapper;
}

AbstractWrapperPtr FuncGraphBuilder::TryToAddNode(const ValuePtr &callable_value,
                                                  const std::vector<AbstractWrapperPtr> &inputs_abstract_wrapper) {
  // Collect the input nodes and input abstracts.
  std::vector<AnfNodePtr> input_node_list;
  std::vector<AbstractBasePtr> input_abs_list;
  if (!GetInputNodesAndAbstracts(callable_value, inputs_abstract_wrapper, &input_node_list, &input_abs_list)) {
    return nullptr;
  }

  CNodePtr new_node;
  AbstractBasePtr abs;
  if (callable_value->isa<Primitive>()) {
    new_node = DoPrimitiveInferAndCheck(callable_value->cast<PrimitivePtr>(), input_node_list, input_abs_list);
    if (new_node != nullptr) {
      abs = new_node->abstract();
    }
  } else {
    // Do infer and check callable.
    abs = DoInferAndCheck(callable_value, input_abs_list);
    if (abs != nullptr) {
      new_node = graph_->NewCNodeInOrder(input_node_list);
    }
  }
  if (new_node == nullptr || abs == nullptr) {
    return nullptr;
  }

  new_node->set_abstract(abs);
  auto ret_abstract_wrapper = std::make_shared<AbstractWrapper>(new_node->abstract());
  (void)key_to_node_.emplace(ret_abstract_wrapper, new_node);
  MS_LOG(INFO) << "Add node: " << new_node->DebugString()
               << " with abstract wrapper: " << ret_abstract_wrapper->ToString();
  return ret_abstract_wrapper;
}

AbstractWrapperPtr FuncGraphBuilder::AddNode(const ValuePtr &callable_value,
                                             const std::vector<AbstractWrapperPtr> &inputs_abstract_wrapper) {
  if (!callable_value->ToAbstract()->isa<abstract::AbstractFunction>()) {
    MS_LOG(INFO) << "The value " << callable_value->ToString() << " is not callable.";
    return nullptr;
  }
  if (callable_value->isa<FuncGraph>()) {
    return AddFgCallNode(callable_value->cast<FuncGraphPtr>(), inputs_abstract_wrapper);
  }
  return TryToAddNode(callable_value, inputs_abstract_wrapper);
}

AbstractWrapperPtr FuncGraphBuilder::AddMultiNode(const std::string &name,
                                                  const std::vector<AbstractWrapperPtr> &inputs_abstract_wrapper) {
  const std::string mod_str = "mindspore.ops.composite.multitype_ops";
  py::module mod = py::module::import(mod_str.c_str());
  if (!py::hasattr(mod, name.c_str())) {
    MS_LOG(INFO) << "Fail to find multitype function graph for name " << name;
    return nullptr;
  }
  py::object fn = mod.attr(name.c_str());
  return AddNode(fn, inputs_abstract_wrapper);
}

bool FuncGraphBuilder::AddOutput(const AbstractWrapperPtr &abstract_wrapper, bool is_top_graph) {
  if (abstract_wrapper == nullptr) {
    return false;
  }
  auto iter = key_to_node_.find(abstract_wrapper);
  if (iter == key_to_node_.end()) {
    MS_LOG(INFO) << "Fail to find correspond anf node for abstract wrapper: " << abstract_wrapper->ToString();
    return false;
  }
  auto node = iter->second;
  MS_EXCEPTION_IF_NULL(node);
  auto abs = node->abstract();
  // Only top graph has restriction on return value type.
  if (is_top_graph && !CheckGraphOutput(abs)) {
    MS_LOG(INFO) << "The output should not be the graph output, abstract: "
                 << (abs == nullptr ? "null" : abs->ToString());
    return false;
  }
  (void)output_nodes_.emplace_back(node);
  return true;
}

FuncGraphPtr FuncGraphBuilder::graph() {
  if (has_set_output_) {
    return graph_;
  }
  if (output_nodes_.empty()) {
    MS_LOG(DEBUG) << "The graph " << graph_->ToString() << " has not been set output.";
    return nullptr;
  }
  bool all_value_node = std::any_of(output_nodes_.begin(), output_nodes_.end(),
                                    [](const AnfNodePtr &node) { return node->isa<ValueNode>(); });
  if (all_value_node) {
    MS_LOG(INFO) << "All graph output is value node, no need to run graph.";
    return nullptr;
  }
  // Single output case.
  if (output_nodes_.size() == 1) {
    graph_->set_output(output_nodes_[0]);
    has_set_output_ = true;
    return graph_;
  }
  // multiple output case.
  output_nodes_.insert(output_nodes_.begin(), NewValueNode(prim::kPrimMakeTuple));
  AbstractBasePtrList abstract_list;
  (void)std::transform(output_nodes_.begin() + 1, output_nodes_.end(), std::back_inserter(abstract_list),
                       [](const AnfNodePtr &node) -> AbstractBasePtr { return node->abstract(); });
  auto output_node = graph_->NewCNodeInOrder(output_nodes_);
  auto fg_output_abs = std::make_shared<abstract::AbstractTuple>(abstract_list);
  output_node->set_abstract(fg_output_abs);

  graph_->set_output(output_node);
  has_set_output_ = true;
  return graph_;
}

void FuncGraphBuilder::ClearNodeAbstract() {
  if (!has_set_output_) {
    MS_LOG(INTERNAL_EXCEPTION) << "Graph not generated, can not clear abstract.";
  }
  static const auto enable_eliminate_unused_element = (common::GetCompileConfig("ENABLE_DDE") != "0");
  auto top_graph = graph();
  if (top_graph == nullptr) {
    return;
  }
  for (const auto &node : mindspore::TopoSort(top_graph->get_return(), SuccDeeperSimple)) {
    MS_EXCEPTION_IF_NULL(node);
    const AbstractBasePtr &prev_inferred = node->abstract();
    auto is_func =
      node->isa<mindspore::ValueNode>() && prev_inferred != nullptr && prev_inferred->isa<abstract::AbstractFunction>();
    // Keep previous inferred value for parameter and ValueNode if the inferred value is not AbstractFunction.
    if (!node->isa<Parameter>() && !is_func) {
      // Reset tuple/list abstract use flags.
      if (enable_eliminate_unused_element && prev_inferred != nullptr &&
          prev_inferred->isa<abstract::AbstractSequence>()) {
        SetSequenceNodeElementsUseFlags(node, nullptr);
      }
      node->set_abstract(nullptr);
      MS_LOG(DEBUG) << "Abstract of node " << node->DebugString() << " is set to nullptr";
    }
  }
}

AbstractWrapperPtr FuncGraphBuilder::AddFgCallNode(const FuncGraphPtr &fg,
                                                   const std::vector<AbstractWrapperPtr> &inputs_abstract_wrapper) {
  std::vector<AnfNodePtr> input_node_list;
  input_node_list.reserve(inputs_abstract_wrapper.size() + 1);

  (void)input_node_list.emplace_back(NewValueNode(fg));
  for (const auto &input_wrapper : inputs_abstract_wrapper) {
    auto node = GetNodeByWrapper(input_wrapper);
    MS_EXCEPTION_IF_NULL(node);
    (void)input_node_list.emplace_back(node);
  }

  auto new_node = graph_->NewCNodeInOrder(input_node_list);
  auto fg_output = fg->output();
  MS_EXCEPTION_IF_NULL(fg_output);
  auto fg_output_abs = fg_output->abstract();
  MS_EXCEPTION_IF_NULL(fg_output_abs);
  new_node->set_abstract(fg_output_abs);

  auto ret_abstract_wrapper = std::make_shared<AbstractWrapper>(new_node->abstract());
  (void)key_to_node_.emplace(ret_abstract_wrapper, new_node);
  return ret_abstract_wrapper;
}

bool FuncGraphBuilder::CheckCallable(const py::object &obj) {
  constexpr auto ms_class_attr = "__ms_class__";
  return py::isinstance<MetaFuncGraph>(obj) ||
         (py::hasattr(obj, PYTHON_PRIMITIVE_FLAG) &&
          parse::data_converter::GetObjType(obj) != parse::RESOLVE_TYPE_CLASS_TYPE) ||
         FunctionShouldBeParseInAst(obj) ||
         (py::hasattr(obj, ms_class_attr) && py::cast<bool>(py::getattr(obj, ms_class_attr)));
}

py::object FuncGraphBuilder::ConvertMethod(const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  py::tuple method_info = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_GET_METHOD_INFO, obj);
  py::object class_name_obj = method_info[0];
  if (py::isinstance<py::none>(class_name_obj)) {
    MS_LOG(INFO) << "Can not get the method info of " << py::str(obj);
    return py::object();
  }
  auto class_name = class_name_obj.cast<std::string>();
  if (class_name == "Tensor" &&
      !py::cast<bool>(python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_IS_MS_TENSOR_METHOD, obj))) {
    return py::object();
  }
  auto type_id = GetTypeIdFromClassName(class_name);
  auto method_name = method_info[1].cast<std::string>();
  MS_LOG(DEBUG) << "type_id: " << type_id << ", method_name: " << method_name;
  Any require = pipeline::Resource::GetMethodPtr(type_id, method_name);
  if (require.empty()) {
    require = pipeline::Resource::GetAttrPtr(type_id, method_name);
  }

  if (require.empty()) {
    MS_LOG(DEBUG) << "Can not find the method registered.";
    return py::object();
  }

  if (require.is<std::string>()) {
    py::function fn = mindspore::python_adapter::GetPyFn(parse::kStandardMethodModelName, require.cast<std::string>());
    if (py::isinstance<py::none>(fn)) {
      MS_LOG(DEBUG) << "Can not find the method '" << require.cast<std::string>() << "' defined in standard_method.";
      return py::object();
    }
    return fn;
  } else if (require.is<PrimitivePtr>()) {
    auto ops_mod = python_adapter::GetPyModule("mindspore.ops");
    auto primitive_class = python_adapter::GetPyObjAttr(ops_mod, "Primitive");
    return primitive_class(require.cast<PrimitivePtr>()->name());
  }
  MS_LOG(DEBUG) << "The method or attr should be a string or a Primitive, but got " << require.ToString();
  return py::object();
}

py::object FuncGraphBuilder::ConvertFunction(const py::object &obj) {
  auto dict = python_adapter::GetPyObjAttr(python_adapter::GetPyModule("mindspore._extends.parse.resources"),
                                           "convert_object_map");
  auto callable_obj_ptr = PyDict_GetItem(dict.ptr(), obj.ptr());
  return callable_obj_ptr == nullptr ? py::object() : py::cast<py::object>(callable_obj_ptr);
}

bool FuncGraphBuilder::CanConstantFoldFunc(const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  py::object can_constant_fold = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_CAN_CONSTANT_FOLD, obj);
  return can_constant_fold.cast<bool>();
}

void FuncGraphBuilder::SetGraphName(const std::string &name) {
  if (name.empty()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph_->debug_info());
  graph_->debug_info()->set_name(name);
}

void FuncGraphBuilder::AddPrevBuilder(const FuncGraphBuilderPtr &builder) { prev_builders_.push_back(builder.get()); }

bool FuncGraphBuilder::ValidateCallableObject(const py::object &obj) {
  if (obj.ptr() == nullptr) {
    return false;
  }
  // Check if object is invalid method for CellList/CellDict, which should not be converted to graph.
  if (CheckInvalidCellListDictMethod(obj)) {
    MS_LOG(INFO) << "The object " << py::str(obj) << " is a invalid CellList/CellDict method, "
                 << "can not convert to graph";
    return false;
  }
  return true;
}

bool FuncGraphBuilder::CheckInvalidCellListDictMethod(const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  py::tuple method_info = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_GET_METHOD_INFO, obj);
  constexpr size_t class_index = 0;
  constexpr size_t method_index = 1;
  py::object class_name_obj = method_info[class_index];
  if (class_name_obj.ptr() == nullptr || py::isinstance<py::none>(class_name_obj)) {
    return false;
  }
  auto class_name = class_name_obj.cast<std::string>();
  MS_LOG(INFO) << "class name: " << class_name;
  if (class_name != "CellList" && class_name != "CellDict") {
    return false;
  }
  auto method_name_obj = method_info[method_index];
  if (method_name_obj.ptr() == nullptr || py::isinstance<py::none>(method_name_obj)) {
    return false;
  }
  auto method_name = method_name_obj.cast<std::string>();
  static std::vector<std::string> inplace_method_name = {"clear", "update"};
  if (std::any_of(inplace_method_name.begin(), inplace_method_name.end(),
                  [&method_name](const std::string &name) { return name == method_name; })) {
    MS_LOG(INFO) << "CellDict/CellList inplace function " << method_name << " found";
    return true;
  }
  auto type_id = GetTypeIdFromClassName(class_name);
  Any require = pipeline::Resource::GetMethodPtr(type_id, method_name);
  return require.empty();
}
}  // namespace mindspore
