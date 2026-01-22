
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
#include "frontend/jit/pi/graph_capture/special_func_infer.h"
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "frontend/jit/pi/runtime.h"
#include "frontend/jit/pi/external.h"
#include "frontend/jit/pi/graph_capture/graph_build.h"
#include "frontend/jit/pi/graph_guard/infer.h"
#include "frontend/jit/pi/graph_capture/side_effect.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"

namespace mindspore {
namespace pijit {
extern ValueNode *GetBoundSelfHelper(CallNode *call_node, bool *is_method);

constexpr const char *kModuleName = "mindspore.graph._pijit.pijit_func_white_list";
constexpr const char *kFuncMapName = "_func_map";
constexpr const char *kSlotCallName = "__call__";
constexpr const size_t kDictPopParamsNum = 2;
constexpr const size_t BoundMethodInputSize = 2;

template <AObject::Type type>
static bool SetCallResType(CallNode *call_node, GraphBuilder *unused = nullptr) {
  call_node->SetVobj(AObject::MakeAObject(type));
  call_node->SetSubGraph(nullptr);
  return false;
}

bool JustCallAndSetRes(CallNode *call_node, GraphBuilder *unused) {
  py::object func = call_node->input(0)->GetVobj()->GetPyObject();
  if (func.ptr() == nullptr) {
    return SetCallResType<AObject::kTypeAnyValue>(call_node);
  }

  std::vector<py::object> args;
  std::transform(call_node->inputs().begin() + 1, call_node->inputs().end(), std::back_inserter(args),
                 [](ValueNode *n) { return n->GetVobj() ? n->GetVobj()->GetPyObject() : py::object(); });
  auto pair = Utils::PackCallStackArgs(args, call_node->GetOpcode(), call_node->kw_names());
  if (pair.first.ptr() == nullptr) {
    return SetCallResType<AObject::kTypeAnyValue>(call_node);
  }

  pi_jit_disable();
  PyObject *value = PyObject_Call(func.ptr(), pair.first.ptr(), pair.second.ptr());
  if (PyErr_Occurred()) {
    MS_LOG(INFO) << "got an error " << py::error_already_set().what() << " at call the "
                 << std::string(py::str(func.ptr()));
    PyErr_Clear();
  }
  pi_jit_enable();

  call_node->SetVobj(AObject::Convert(value));
  call_node->SetSubGraph(nullptr);
  Py_XDECREF(value);
  return false;
}

bool JustCallAndSetResWithArgs(CallNode *call_node, const std::vector<py::object> &args, GraphBuilder *unused) {
  py::object func = call_node->input(0)->GetVobj()->GetPyObject();
  if (func.ptr() == nullptr) {
    return SetCallResType<AObject::kTypeAnyValue>(call_node);
  }

  auto pair = Utils::PackCallStackArgs(args, call_node->GetOpcode(), call_node->kw_names());
  if (pair.first.ptr() == nullptr) {
    return SetCallResType<AObject::kTypeAnyValue>(call_node);
  }

  pi_jit_disable();
  PyObject *value = PyObject_Call(func.ptr(), pair.first.ptr(), pair.second.ptr());
  if (PyErr_Occurred()) {
    MS_LOG(INFO) << "got an error " << py::error_already_set().what() << " at call the "
                 << std::string(py::str(func.ptr()));
    PyErr_Clear();
  }
  pi_jit_enable();

  call_node->SetVobj(AObject::Convert(value));
  call_node->SetSubGraph(nullptr);
  Py_XDECREF(value);
  return false;
}

static bool CallNodeReturnConst(CallNode *call_node, Graph *sub_graph, AObject *value) {
  PyObject *cnst = value->GetPyObject().ptr();
  MS_EXCEPTION_IF_NULL(cnst);

  ValueNode *ret_node = sub_graph->NewValueNode(value, LOAD_CONST, -1, {});
  ret_node->SetGraph(call_node->GetGraph());
  sub_graph->SetRetVal(ret_node);
  call_node->SetVobj(sub_graph->GetRetVal()->GetVobj());
  call_node->SetSubGraph(sub_graph);
  call_node->SetInlineReason(InlineReason::kInline);
  return true;
}

static bool InferGetCachePrim(CallNode *n, GraphBuilder *unused = nullptr) {
  // just return the first parameter of _get_cache_prim
  Graph *g = n->GetSubGraph();
  n->SetVobj(n->input(1)->GetVobj());
  g->SetRetVal(n->input(1));
  return true;
}

static bool InferRegistryGet(CallNode *call_node, GraphBuilder *unused = nullptr) {
  Graph *g = call_node->GetSubGraph();
  JustCallAndSetRes(call_node);

  py::object func = call_node->GetVobj()->GetPyObject();
  if (call_node->inputs().back()->GetOpcode() == LOAD_CONST && func.ptr() != nullptr) {
    return CallNodeReturnConst(call_node, g, call_node->GetVobj());
  }
  return false;
}

static bool GuardBuiltinFunc(CallNode *call_node) {
  auto func_node = call_node->input(0);
  MS_EXCEPTION_IF_NULL(func_node);
  if (func_node->GetVobj() == nullptr) {
    return false;
  }
  PyObject *func = func_node->GetVobj()->GetPyObject().ptr();
  if (PyMethod_Check(func)) {
    auto self = PyMethod_GET_SELF(func);
    if (IsTensorType<true>(Py_TYPE(self))) {
      auto self_node = call_node->GetSelf();
      if (self_node == nullptr) {
        MS_LOG(WARNING) << "failed to find self value node for call node" << call_node->ToString();
        return false;
      }
      if (self_node->GetOpcode() != LOAD_CONST) {
        auto self_node_wrapper = self_node->abstract_wrapper();
        if (self_node_wrapper == nullptr) {
          MS_LOG(WARNING) << "Failed to find wrapper for tensor self, node: " << call_node->ToString();
          return false;
        }
        if (!self_node_wrapper->IsConstant()) {
          MS_LOG(INFO) << "Fail to guard built-in function due to variable self tensor " << call_node->ToString();
          return false;
        }
      }
    }
  }
  Graph *graph = call_node->GetGraph();
  MS_EXCEPTION_IF_NULL(graph);
  bool guard_inputs = call_node->GetVobj()->GetType() == AObject::kTypeAnyValue;
  const auto &call_node_inputs = call_node->inputs();
  for (size_t i = 1; i < call_node_inputs.size(); ++i) {
    auto cur_input = call_node_inputs[i];
    MS_EXCEPTION_IF_NULL(cur_input);
    if (cur_input->GetOpcode() == LOAD_CONST) {
      continue;
    }
    if (cur_input->GetVobj()->GetType() == AObject::kTypeTensor) {
      auto cur_input_wrapper = cur_input->abstract_wrapper();
      if (cur_input_wrapper == nullptr) {
        MS_LOG(WARNING) << "Failed to guard built-in function since wrapper is nullptr for " << cur_input->ToString();
        return false;
      }
      if (!cur_input_wrapper->IsConstant()) {
        MS_LOG(INFO) << "Failed to guard built-in function due to variable input " << cur_input->ToString();
        return false;
      }
    }
    if (guard_inputs && !graph->GuardValueNode(cur_input)) {
      return false;
    }
  }
  return guard_inputs ? true : graph->GuardValueNode(call_node);
}

static bool GuardIsInstance(CallNode *call_node) {
  Graph *graph = call_node->GetGraph();
  const auto &cnst = call_node->input(1)->GetConstantInfo();
  if (cnst != nullptr && cnst->type() != nullptr) {
    constexpr int second_arg = 2;
    auto success = graph->GuardValueNode(call_node->input(second_arg));
    if (!success && (call_node->GetGraph()->Config().getIntConfig(GraphJitConfig::kGuardRelaxCount) > 0)) {
      TracePtr tr = graph->TraceValueNode(call_node->input(second_arg));
      if (tr == nullptr) {
        return true;
      }
    }
    return success;
  }
  auto success = graph->GuardValueNode(call_node);
  if (!success && (call_node->GetGraph()->Config().getIntConfig(GraphJitConfig::kGuardRelaxCount) > 0)) {
    TracePtr tr = graph->TraceValueNode(call_node);
    if (tr == nullptr) {
      return true;
    }
  }
  return success;
}

bool InferBuiltinFuncOrMethod(CallNode *call_node, GraphBuilder *unused = nullptr) {
  bool guard_success = false;
  std::string name = GetFuncName(call_node->input(0)->GetVobj()->GetPyObject());
  if (name == "isinstance") {
    guard_success = GuardIsInstance(call_node);
  } else {
    guard_success = GuardBuiltinFunc(call_node);
  }

  if (!guard_success) {
    MS_LOG(INFO) << "Guard `isinstance` or guard builtin func failed.";
    call_node->SetSubGraph(nullptr);
    return false;
  }

  Graph *sub_graph = call_node->GetSubGraph();
  (void)JustCallAndSetRes(call_node);
  call_node->SetInlineReason(InlineReason::kInlineFunc_Type_Unsupported);
  ConstantInfo::CollectBuiltinFuncConstantInfo(call_node);
  if (call_node->IsConstantValue()) {
    return CallNodeReturnConst(call_node, sub_graph, call_node->GetVobj());
  }
  if (call_node->GetVobj() == nullptr || call_node->GetVobj()->GetPyObject().ptr() == nullptr) {
    return false;
  }
  return CallNodeReturnConst(call_node, sub_graph, call_node->GetVobj());
}

// dict.items()
bool InferDictItems(CallNode *call_node, GraphBuilder *builder) {
  MS_LOG(INFO) << "Start to handle dict items.";
  MS_EXCEPTION_IF_NULL(builder);
  (void)JustCallAndSetRes(call_node);
  auto func = call_node->input(0);
  if (func->GetOpcode() == LOAD_ATTR) {
    auto dict_node = func->input(0);
    MS_EXCEPTION_IF_NULL(dict_node);
    auto wrapper = dict_node->abstract_wrapper();
    if (wrapper == nullptr) {
      MS_LOG(INFO) << "Wrapper is NULL for dict node: " << dict_node->ToString() << ", failed to infer dict.items()";
      return false;
    }
    AbstractWrapperPtrList inputs_wrapper = {wrapper};
    auto ret = builder->FGBuilder()->AddNode(prim::kPrimDictItems, inputs_wrapper);
    if (ret == nullptr) {
      MS_LOG(INFO) << "Handle dict items failed for node: " << call_node->ToString();
      return false;
    }
    call_node->set_abstract_wrapper(ret);
    return true;
  }
  MS_LOG(INFO) << "Failed to infer dict.items for node: " << call_node->ToString();
  return false;
}

static void RecordBuiltinMethodSideEffect(Graph *graph, CallNode *call_node, const std::string &method_name) {
  const auto &side_effect = graph->GetSideEffect();
  side_effect->Record(call_node, SideEffect::kBuiltinMethod, method_name);
}

static bool InferListAppend(CallNode *call_node, GraphBuilder *parent) {
  call_node->SetSubGraph(nullptr);

  // check is supported type and get arguments
  bool is_method_descriptor = false;
  ValueNode *self = GetSelfFromKnownMethod(call_node, &is_method_descriptor);
  if (self == nullptr) {
    return false;
  }
  ValueNode *new_element = call_node->input(1 + is_method_descriptor);

  // transform to "new_list = [old_list[0], old_list[1]..., new_element]"
  int size = parent->frame().GetStacks().size();
  if (!parent->UnpackElements(self)) {
    return false;
  }
  parent->push(new_element);
  size = parent->frame().GetStacks().size() - size;
  parent->DoBuildOp({BUILD_LIST, size});
  auto new_node = parent->pop();
  auto old_node = self;
  old_node->GetVobj()->SetNextVersion(new_node->GetVobj());

  // constant fold and set node info
  // todo: this builder do not need to create.
  auto builder = std::make_shared<GraphBuilder>(parent->root(), parent, nullptr, nullptr);
  Graph *sub_graph = builder->GetGraph();
  builder->DoLoadConst({LOAD_CONST, 0, py::object(py::none())});
  builder->DoReturn({RETURN_VALUE, 0});

  call_node->SetSubGraph(sub_graph);
  call_node->SetVobj(sub_graph->GetRetVal()->GetVobj());
  call_node->SetInlineReason(InlineReason::kInline);

  // update frame status and record side-effect
  bool is_referenced = false;
  parent->ReplaceAll(old_node, new_node, &is_referenced);
  const auto &replace_map = parent->GetGraph()->GetSideEffect()->data()->modified_and_replaced_map();
  auto is_param_expander_enable = parent->GetGraph()->Config().GetBoolConfig(GraphJitConfig::kExpandGraphInput);
  bool is_new_var =
    self->GetOpcode() == BUILD_LIST && replace_map.find(self) == replace_map.end() && !is_param_expander_enable;
  if (!is_new_var || is_referenced || self == new_element) {
    parent->GetGraph()->GetSideEffect()->data()->RecordModifiedAndReplacedNode(old_node, new_node);
    RecordBuiltinMethodSideEffect(parent->GetGraph(), call_node, "append");
  }
  return true;
}

template <typename ElementsAction, typename FuncReturnAction, typename SafeReplaceChecker>
static bool InferListMethodWithSideEffect(CallNode *call_node, GraphBuilder *parent, ElementsAction action,
                                          FuncReturnAction return_action, SafeReplaceChecker is_safe_replace,
                                          const std::string &method_name) {
  call_node->SetSubGraph(nullptr);

  // check is supported type and get arguments
  bool is_method_descriptor = false;
  ValueNode *self = GetSelfFromKnownMethod(call_node, &is_method_descriptor);
  if (self == nullptr) {
    return false;
  }
  // transform to "new_list = [old_list[0], old_list[1]..., new_element]"
  const auto &stack = parent->frame().GetStacks();
  int size = stack.size();
  if (!parent->UnpackElements(self)) {
    return false;
  }
  size = stack.size() - size;
  std::vector<ValueNode *> elements(stack.end() - size, stack.end());
  parent->popn(size);
  // elements actions
  action(call_node, parent, &elements);
  for (const auto &i : elements) {
    parent->push(i);
  }
  size = elements.size();
  parent->DoBuildOp({BUILD_LIST, size});
  // set function return value
  return_action(call_node, parent);

  // update frame status and record side-effect
  auto new_node = parent->pop();
  auto old_node = self;
  old_node->GetVobj()->SetNextVersion(new_node->GetVobj());
  bool is_referenced = false;
  parent->ReplaceAll(old_node, new_node, &is_referenced);
  const auto &replace_map = parent->GetGraph()->GetSideEffect()->data()->modified_and_replaced_map();
  auto is_param_expander_enable = parent->GetGraph()->Config().GetBoolConfig(GraphJitConfig::kExpandGraphInput);
  bool is_new_var =
    self->GetOpcode() == BUILD_LIST && replace_map.find(self) == replace_map.end() && !is_param_expander_enable;
  if (!is_new_var || is_referenced || !is_safe_replace(call_node, parent)) {
    parent->GetGraph()->GetSideEffect()->data()->RecordModifiedAndReplacedNode(old_node, new_node);
    RecordBuiltinMethodSideEffect(parent->GetGraph(), call_node, method_name);
  }
  return true;
}

static bool InferListReverse(CallNode *call_node, GraphBuilder *parent) {
  auto reverse_action = [](CallNode *, GraphBuilder *, std::vector<ValueNode *> *elements) {
    std::reverse(elements->begin(), elements->end());
  };
  auto return_none = [](CallNode *call_node, GraphBuilder *parent) {
    // todo: this builder do not need to create.
    auto builder = std::make_shared<GraphBuilder>(parent->root(), parent, nullptr, nullptr);
    CallNodeReturnConst(call_node, builder->GetGraph(), AObject::Convert(Py_None));
  };
  auto is_safe = [](CallNode *, GraphBuilder *) { return true; };
  return InferListMethodWithSideEffect(call_node, parent, reverse_action, return_none, is_safe, "reverse");
}

static bool InferListPop(CallNode *call_node, GraphBuilder *parent) {
  call_node->SetSubGraph(nullptr);
  call_node->SetInlineReason(InlineReason::kInlineFunc_Type_Unsupported);
  if (call_node->GetOpcode() == CALL_FUNCTION_EX) {
    return false;
  }
  Py_ssize_t index = -1;
  if (call_node->GetOparg() != 0) {  // tack exactly one arg
    bool is_method_descriptor = false;
    (void)GetSelfFromKnownMethod(call_node, &is_method_descriptor);
    auto index_node = call_node->input(1 + is_method_descriptor);
    if (!parent->GetGraph()->GuardValueNode(index_node)) {
      return false;
    }
    // only accept pop a constant index
    index = py::int_(index_node->GetVobj()->GetPyObject());
  }

  ValueNode *pop_value = nullptr;
  auto pop_action = [&pop_value, &index](CallNode *call_node, GraphBuilder *, std::vector<ValueNode *> *elements) {
    index = index < 0 ? static_cast<Py_ssize_t>(elements->size()) + index : index;
    auto iter = elements->begin() + index;
    pop_value = *iter;
    (void)elements->erase(iter);
    return true;
  };
  auto return_element = [&pop_value](CallNode *call_node, GraphBuilder *parent) {
    MS_EXCEPTION_IF_NULL(pop_value);
    // todo: this builder do not need to create.
    auto builder = std::make_shared<GraphBuilder>(parent->root(), parent, nullptr, nullptr);
    auto sub_graph = builder->GetGraph();
    sub_graph->SetRetVal(pop_value);
    call_node->SetSubGraph(sub_graph);
    call_node->SetVobj(sub_graph->GetRetVal()->GetVobj());
    call_node->SetInlineReason(InlineReason::kInline);
  };
  auto is_safe = [](CallNode *, GraphBuilder *) { return true; };
  return InferListMethodWithSideEffect(call_node, parent, pop_action, return_element, is_safe, "pop");
}

static bool InferListRemove(CallNode *call_node, GraphBuilder *parent) {
  call_node->SetSubGraph(nullptr);
  call_node->SetInlineReason(InlineReason::kInlineFunc_Type_Unsupported);
  if (call_node->GetOpcode() == CALL_FUNCTION_EX) {
    return false;
  }
  bool is_descr = false;
  ValueNode *self = GetSelfFromKnownMethod(call_node, &is_descr);
  if (self == nullptr) {
    return false;
  }
  ValueNode *target = call_node->input(1 + is_descr);
  const auto &elem = self->inputs();
  if (self->GetOpcode() != BUILD_LIST || elem.end() == std::find(elem.begin(), elem.end(), target)) {
    return false;  // erase any value
  }
  /**
   * only specialized for this case that object id is find in list:
   * my_list = list_build(x, y, z)
   * my_list.erase(x)
   */
  auto remove_action = [&is_descr, &target](CallNode *call_node, GraphBuilder *, std::vector<ValueNode *> *elements) {
    (void)std::remove(elements->begin(), elements->end(), target);
  };
  auto return_none = [](CallNode *call_node, GraphBuilder *parent) {
    // todo: this builder do not need to create.
    auto builder = std::make_shared<GraphBuilder>(parent->root(), parent, nullptr, nullptr);
    CallNodeReturnConst(call_node, builder->GetGraph(), AObject::Convert(Py_None));
  };
  auto is_safe = [](CallNode *, GraphBuilder *) { return true; };
  return InferListMethodWithSideEffect(call_node, parent, remove_action, return_none, is_safe, "remove");
}

static bool InferDictPop(CallNode *call_node, GraphBuilder *parent) {
  call_node->SetSubGraph(nullptr);

  bool is_method_descriptor = false;
  ValueNode *self = GetSelfFromKnownMethod(call_node, &is_method_descriptor);
  if (self == nullptr) {
    return false;
  }
  // guard dict key and convert to constant key map
  if (!parent->GetGraph()->GuardValueNode(self)) {
    return false;
  }

  ValueNode *dict_node = self;
  ValueNode *key_node = call_node->input(1 + is_method_descriptor);
  ValueNode *default_node = call_node->inputs().size() > (kDictPopParamsNum + is_method_descriptor)
                              ? call_node->input(kDictPopParamsNum + is_method_descriptor)
                              : nullptr;
  // get key from dict
  py::object dict = dict_node->GetVobj()->GetPyObject();
  py::object key = key_node->GetVobj()->GetPyObject();
  MS_EXCEPTION_IF_CHECK_FAIL(PyDict_Check(dict.ptr()), "for dict.pop, first parameter must be a dict");
  py::object value = py::reinterpret_borrow<py::object>(PyDict_GetItem(dict.ptr(), key.ptr()));
  if (value.ptr() == nullptr) {
    if (default_node == nullptr) {
      return false;  // key error
    }
    value = default_node->GetVobj()->GetPyObject();
  }

  // transform to "new_map = {key:old_map[key]...}"
  ValueNode *old_node = dict_node;
  ValueNode *new_node = parent->TransformDictSetItem(dict_node, key_node, nullptr, default_node != nullptr);
  if (new_node == nullptr) {
    MS_LOG(INFO) << "Failed to infer dict.pop() for node: " << call_node->ToString();
    return false;
  }

  // constant fold and set node info
  // todo: this builder do not need to create.
  auto builder = std::make_shared<GraphBuilder>(parent->root(), parent, nullptr, nullptr);
  Graph *sub_graph = builder->GetGraph();
  builder->DoLoadConst({LOAD_CONST, 0, value});
  builder->DoReturn({RETURN_VALUE, 0});

  call_node->SetSubGraph(sub_graph);
  call_node->SetVobj(sub_graph->GetRetVal()->GetVobj());
  call_node->SetInlineReason(InlineReason::kInline);

  // update frame status and record side-effect
  bool is_referenced = false;
  parent->ReplaceAll(old_node, new_node, &is_referenced);
  const auto &replace_map = parent->GetGraph()->GetSideEffect()->data()->modified_and_replaced_map();
  auto is_param_expander_enable = parent->GetGraph()->Config().GetBoolConfig(GraphJitConfig::kExpandGraphInput);
  bool is_new_var =
    self->GetOpcode() == BUILD_MAP && replace_map.find(self) == replace_map.end() && !is_param_expander_enable;
  if (!is_new_var || is_referenced) {
    parent->GetGraph()->GetSideEffect()->data()->RecordModifiedAndReplacedNode(old_node, new_node);
    RecordBuiltinMethodSideEffect(parent->GetGraph(), call_node, "pop");
  }
  return true;
}

static bool SetForbiddenFuncInfo(CallNode *call_node, GraphBuilder *unused = nullptr) {
  SetCallResType<AObject::kTypeAnyValue>(call_node);
  call_node->SetInlineReason(InlineReason::kInlineFunc_Type_Unsupported);
  return false;
}

static void TensorAssignValue(CallNode *call_node, GraphBuilder *parent, ValueNode *old_value, ValueNode *new_value,
                              SideEffect::Type type, const char *name, bool need_copy = true) {
  auto old_node = old_value;
  auto new_node = need_copy ? parent->MakeTensorCopy(new_value) : new_value;

  call_node->SetSubGraph(nullptr);
  old_value->GetVobj()->SetNextVersion(new_node->GetVobj());
  call_node->SetVobj(new_node->GetVobj());

  // update frame status and record side-effect
  bool is_referenced = false;
  parent->ReplaceAll(old_node, new_node, &is_referenced);
  if (call_node != new_node) {
    parent->ReplaceAll(call_node, new_node, &is_referenced);
  }
  is_referenced = IsReferencedVariable(old_value);
  MS_LOG(INFO) << "check the node is referenced: " << is_referenced << " [" << old_value->ToString();
  if (!is_referenced) {
    // a new local variable and it's not referenced, modify operations is not side effect
    // just replaced old_value by new_value and remove modify operations
    return;
  }
  parent->GetGraph()->GetSideEffect()->data()->RecordModifiedAndReplacedNode(old_node, new_node);
  parent->GetGraph()->GetSideEffect()->Record(call_node, type, name);
}

bool InferTensorAssignValue(CallNode *call_node, GraphBuilder *parent) {
  bool is_not_method = false;
  ValueNode *self = GetSelfFromKnownMethod(call_node, &is_not_method);
  if (self == nullptr || !Opcode(call_node->GetOpcode()).IsCallFunc()) {
    call_node->SetSubGraph(nullptr);
    return false;
  }
  TensorAssignValue(call_node, parent, self, call_node->input(1 + is_not_method), SideEffect::kBuiltinMethod,
                    "assign_value");
  return true;
}

bool InferPrimitiveAssign(CallNode *call_node, GraphBuilder *parent) {
  TensorAssignValue(call_node, parent, call_node->input(1), call_node->input(kTwo), SideEffect::kDefault, "");
  return true;
}

bool InferTensorSetItem(CallNode *call_node, GraphBuilder *builder) {
  SetForbiddenFuncInfo(call_node);
  bool is_not_method = false;
  ValueNode *self = GetSelfFromKnownMethod(call_node, &is_not_method);
  if (self == nullptr) {
    return false;
  }
  if (self->GetVobj() == nullptr || self->GetVobj()->GetType() != AObject::kTypeTensor) {
    return false;
  }

  py::object method = FuncGraphBuilder::ConvertMethod("Tensor", "__setitem__");
  MS_EXCEPTION_IF_NULL(method.ptr());
  std::vector<ValueNode *> args = {self};
  for (size_t i = 1 + is_not_method; i < call_node->inputs().size(); ++i) {
    args.push_back(call_node->input(i));
  }
  StopTraceReason stop_reason = StopTraceReason::kNonStopTrace;
  builder->FGAddNode(call_node, method, builder->HandleInputArgs(std::move(args)), &stop_reason);
  if (!call_node->has_abstract_wrapper()) {
    MS_LOG(INFO) << "Tensor setitem infer failed";
    return false;
  }

  TensorAssignValue(call_node, builder, self, call_node, SideEffect::kBuiltinMethod, "__setitem__", false);

  call_node->SetInlineReason(InlineReason::kInlineFuncSpecialize);
  return true;
}

bool InferTensorIsContiguous(CallNode *call_node, GraphBuilder *) {
  CallNodeReturnConst(call_node, call_node->GetSubGraph(), AObject::Convert(Py_False));
  return true;
}

enum FuncKey {
  FUNC_KEY_EMPTY = 0,             // ""
  FUNC_KEY_PIJIT_CONSTEXPR,       // "pijit.constexpr"
  FUNC_KEY_PIJIT_FORBIDDEN,       // "pijit.forbidden"
  FUNC_KEY_BUILTIN_FUNC,          // "builtin.func"
  FUNC_KEY_LIST_APPEND,           // "list.append"
  FUNC_KEY_DICT_POP,              // "dict.pop"
  FUNC_KEY_PRIMITIVE,             // "mindspore._c_expression.Primitive_"
  FUNC_KEY_META_FUNCG_RAPH,       // "mindspore._c_expression.MetaFuncGraph_"
  FUNC_KEY_PSJIT_CODE,            // "mindspore.graph.api.jit.<locals>.staging_specialize"
  FUNC_KEY_CONSTEXPR,             // "mindspore.ops.primitive.constexpr"
  FUNC_KEY_PRIMEXPR,              // "mindspore.ops.primitive._primexpr"
  FUNC_KEY_GET_CACHE_PRIM,        // "mindspore.ops._primitive_cache._get_cache_prim"
  FUNC_KEY_REGISTRY_GET,          // "mindspore.common._register_for_tensor.Registry.get"
  FUNC_KEY_TENSOR_ASTYPE,         // "mindspore.common.tensor.Tensor.astype"
  FUNC_KEY_GRAD_OPERATIONS_CODE,  // "mindspore.ops.composite.base._Grad.__call__.<locals>.after_grad"
  FUNC_KEY_PSJIT_CONVERTMAP,      // "mindspore.graph._parse.resources.convert_object_map"
  FUNC_KEY_GRAPH_CELL,            // "mindspore.nn.cell.GraphCell"
  FUNC_KEY_MS_API,                // mindspore api
  FUNC_KEY_MAPPING_GET,           // mapping get
  FUNC_KEY_LIST_POP,              // list.pop
  FUNC_KEY_LIST_REMOVE,           // list.remove
  FUNC_KEY_LIST_REVERSE,          // list.reverse
  FUNC_KEY_DICT_ITEMS,            // dict.items
  FUNC_KEY_PRIMITIVE_ASSIGN,      // mindspore.ops.assign, Primitive("Assign")
  FUNC_KEY_TENSOR_SETITEM,        // Tensor.__setitem__
  FUNC_KEY_TENSOR_ASSIGN_VALUE,   // Tensor.assign_value
  FUNC_KEY_TENSOR_IS_CONTIGUOUS,  // Tensor.is_contiguous
  FUNC_KEY_COUNT,
};
static FuncKey FindFuncKey(const py::object &callable);

static const std::unordered_map<FuncKey, InferFunc> infer_func_map = {
  {FUNC_KEY_PIJIT_CONSTEXPR, JustCallAndSetRes},
  {FUNC_KEY_PIJIT_FORBIDDEN, SetForbiddenFuncInfo},
  {FUNC_KEY_LIST_APPEND, InferListAppend},
  {FUNC_KEY_DICT_POP, InferDictPop},
  {FUNC_KEY_BUILTIN_FUNC, InferBuiltinFuncOrMethod},
  {FUNC_KEY_PSJIT_CODE, SetCallResType<AObject::kTypeTensor>},
  {FUNC_KEY_GET_CACHE_PRIM, InferGetCachePrim},
  {FUNC_KEY_REGISTRY_GET, InferRegistryGet},
  {FUNC_KEY_LIST_POP, InferListPop},
  {FUNC_KEY_LIST_REMOVE, InferListRemove},
  {FUNC_KEY_LIST_REVERSE, InferListReverse},
  {FUNC_KEY_DICT_ITEMS, InferDictItems},
  {FUNC_KEY_PRIMITIVE_ASSIGN, InferPrimitiveAssign},
  {FUNC_KEY_TENSOR_SETITEM, InferTensorSetItem},
  {FUNC_KEY_TENSOR_ASSIGN_VALUE, InferTensorAssignValue},
  {FUNC_KEY_TENSOR_IS_CONTIGUOUS, InferTensorIsContiguous},
};

InferFunc FindInferFunc(const py::object &callable) {
  FuncKey k = FindFuncKey(callable);
  auto iter = infer_func_map.find(k);
  if (iter != infer_func_map.end()) {
    return iter->second;
  }
  return nullptr;
}

static const std::unordered_map<size_t, FuncKey> &GetFuncKeyMap() {
  static std::unordered_map<size_t, FuncKey> map = {};
  if (!map.empty()) {
    return map;
  }
  py::object func_map = Utils::GetModuleAttr(kModuleName, kFuncMapName, true, true);
  MS_EXCEPTION_IF_CHECK_FAIL(PyDict_CheckExact(func_map.ptr()), "white list func map must be 'dict[int, int]'");
  PyObject *key;
  PyObject *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(func_map.ptr(), &pos, &key, &value)) {
    MS_EXCEPTION_IF_CHECK_FAIL(PyLong_CheckExact(key), "white list func map key must be 'int'");
    MS_EXCEPTION_IF_CHECK_FAIL(PyLong_CheckExact(value), "white list func map value must be 'int'");
    size_t k = (PyLong_AsSize_t(value));
    MS_EXCEPTION_IF_CHECK_FAIL(k < FUNC_KEY_COUNT, "white list func map got error FuncKey " + std::to_string(k));
    map[PyLong_AsSize_t(key)] = static_cast<FuncKey>(k);
  }
  return map;
}

static FuncKey KeyFinderFuncId(const py::object &callable) {
  auto iter = GetFuncKeyMap().find(FunctionId(callable));
  return iter != GetFuncKeyMap().end() ? iter->second : FUNC_KEY_EMPTY;
}

static FuncKey KeyFinderFuncCodeId(const py::object &callable) {
  PyObject *func = callable.ptr();
  py::object handle;
  if (IsCellType<true>(Py_TYPE(func))) {
    handle = callable.attr("construct");
    func = handle.ptr();
  }
  if (PyMethod_Check(func)) {
    func = PyMethod_GET_FUNCTION(func);
  }
  if (PyFunction_Check(func)) {
    func = PyFunction_GET_CODE(func);
  }
  if (!PyCode_Check(func)) {
    return FUNC_KEY_EMPTY;
  }
  auto iter = GetFuncKeyMap().find(reinterpret_cast<size_t>(func));
  return iter != GetFuncKeyMap().end() ? iter->second : FUNC_KEY_EMPTY;
}

static FuncKey KeyFinderPrimitive(const py::object &callable) {
  PyTypeObject *type_object = Py_TYPE(callable.ptr());
  bool convert_to_prim = IsPrimitiveType<true>(type_object) || IsPrimitiveFunctionType<true>(type_object);
  if (!convert_to_prim) {
    return FUNC_KEY_EMPTY;
  }
  py::object func = py::getattr(reinterpret_cast<PyObject *>(type_object), kSlotCallName, nullptr);
  size_t id;
  if (func.ptr() == nullptr) {
    // primitive not defined slot __call__, use it self as id
    id = reinterpret_cast<size_t>(callable.ptr());
  } else if (PyFunction_Check(func.ptr())) {
    // primitive defined python function __call__
    id = reinterpret_cast<size_t>(PyFunction_GET_CODE(func.ptr()));
  } else {
    // primitive defined cpp function __call__
    id = FunctionId(func);
  }
  // first, find map to check special primitive.
  auto iter = GetFuncKeyMap().find(id);
  return iter != GetFuncKeyMap().end() ? iter->second : FUNC_KEY_PRIMITIVE;
}

static size_t GetGraphCellTypeId() {
  static size_t graph_cell_type_id = 0;
  if (graph_cell_type_id == 0) {
    py::object type = Utils::GetModuleAttr("mindspore.nn.cell", "GraphCell", false, true);
    graph_cell_type_id = reinterpret_cast<size_t>(type.ptr());
  }
  return graph_cell_type_id;
}

static FuncKey KeyFinderCallableType(const py::object &callable) {
  PyTypeObject *type_object = reinterpret_cast<PyTypeObject *>(callable.ptr());
  type_object = PyType_CheckExact(type_object) ? type_object : Py_TYPE(type_object);
  size_t type_id = reinterpret_cast<size_t>(type_object);
  if (IsPrimitiveType<true>(type_object) || IsPrimitiveFunctionType<true>(type_object)) {
    return KeyFinderPrimitive(callable);
  } else if (IsMetaFuncGraphType<true>(type_object)) {
    return FUNC_KEY_META_FUNCG_RAPH;
  } else if (type_id == GetGraphCellTypeId()) {
    return FUNC_KEY_GRAPH_CELL;
  }
  return FUNC_KEY_EMPTY;
}

static FuncKey KeyFinderSkipModule(const py::object &callable) {
  const auto &modules = kPIJitConfigDefault.allowed_inline_modules();
  std::string mod = GetTopModule(callable);
  if (modules.find(mod) != modules.end()) {
    return FUNC_KEY_EMPTY;
  }

  PyObject *func_info = callable.ptr();
  if (PyMethod_Check(func_info)) {
    func_info = PyMethod_GET_FUNCTION(func_info);
  }
  if (!PyFunction_Check(func_info) && !PyCFunction_Check(func_info) && !PyType_Check(func_info)) {
    func_info = reinterpret_cast<PyObject *>(Py_TYPE(func_info));
  }
  if (IsPiJitLogOn(LogCfg::kGraphBreak)) {
    MS_LOG(ERROR) << "func " << std::string(py::str(func_info)) << " is forbidden to analyze, module is " << mod;
  }
  return FUNC_KEY_PIJIT_FORBIDDEN;
}

static FuncKey FindFuncKey(const py::object &callable) {
  static std::vector<FuncKey (*)(const py::object &callable)> finders = {
    KeyFinderFuncId, KeyFinderFuncCodeId, KeyFinderCallableType, KeyFinderSkipModule,  // must be last for check modules
  };
  if (callable.ptr() == nullptr || !PyCallable_Check(callable.ptr())) {
    return FUNC_KEY_EMPTY;
  }
  FuncKey res = FUNC_KEY_EMPTY;
  for (auto iter = finders.begin(), end = finders.end(); iter != end && res == FUNC_KEY_EMPTY; ++iter) {
    res = (*iter)(callable);
  }
  return res;
}

bool IsPSJitFunction(const py::object &callable_info) {
  if (callable_info.ptr() == nullptr) {
    return false;
  }
  if (FindFuncKey(callable_info) == FUNC_KEY_PSJIT_CODE) {
    return true;
  }
  return false;
}

bool CheckJitConstexpr(const py::object &func) {
  if (func.ptr() == nullptr) {
    return false;
  }
  FuncKey k = KeyFinderFuncId(func);
  return k == FUNC_KEY_PIJIT_CONSTEXPR;
}

bool CheckMSConstexpr(const py::object &func) {
  if (func.ptr() == nullptr) {
    return false;
  }
  FuncKey k = KeyFinderPrimitive(func);
  return k == FUNC_KEY_CONSTEXPR || k == FUNC_KEY_PRIMEXPR;
}

bool CheckBuiltinFuncOrMethod(const py::object &func) {
  if (func.ptr() == nullptr) {
    return false;
  }
  FuncKey k = KeyFinderFuncId(func);
  return k == FUNC_KEY_BUILTIN_FUNC;
}

static bool IsNewVariable(ValueNode *node) {
  Opcode op(node->GetOpcode());
  if ((op == BINARY_SUBSCR || op.IsBinaryMath()) && node->input(0)->GetVobj()->GetType() == AObject::kTypeTensor) {
    // specialization of Tensor
    return true;
  }
  if (op.IsBuildOp() || (op.IsBinaryMath() && op.MayDelete()) || op.IsUnaryMath()) {
    // builtin type create, binary math without inplace, unary math
    return true;
  }
  if (!op.IsCall()) {
    // unknown source or operations
    return false;
  }
  AObject::Type callable_type = node->input(0)->GetVobj()->GetType();
  AObject::Type result_type = node->GetVobj() ? node->GetVobj()->GetType() : AObject::kTypeAnyValue;
  if (callable_type == AObject::kTypeType) {
    // type call, create a new object
    return true;
  }
  if (node->GetType() != ValueNode::Call) {
    return false;
  }
  CallNode *cn = static_cast<CallNode *>(node);
  if (cn->GetSubGraph() != nullptr && cn->GetSubGraph()->GetRetVal() != nullptr) {
    // resusive check function return value
    return IsNewVariable(cn->GetSubGraph()->GetRetVal());
  }
  if (result_type == AObject::kTypeTensor) {
    // tensor operations always create a new Tensor
    return true;
  }
  return false;
}

static bool CheckReferenced(Graph *graph, ValueNode *target) {
  for (auto maybe_ref : graph->GetTracedNodes()) {
    Opcode op(maybe_ref->GetOpcode());
    if (op.MayDelete()) {
      continue;  // only read the variable
    }
    const auto &used = maybe_ref->inputs();
    if (used.end() == std::find(used.begin(), used.end(), target)) {
      continue;  // not used target
    }
    if (op.IsBuildOp()) {
      // the target is a elements of container, check container reference and liveness when optimize sideeffect
      return true;
    }
    if (maybe_ref->GetType() == ValueNode::Call) {
      Graph *sub_graph = static_cast<CallNode *>(maybe_ref)->GetSubGraph();
      if (sub_graph == nullptr || CheckReferenced(sub_graph, target)) {
        return true;
      }
      continue;
    }
    return true;
  }
  return false;
}

bool IsReferencedVariable(ValueNode *target) {
  if (!IsNewVariable(target)) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(target->GetGraph());
  const auto &replace_map = target->GetGraph()->GetSideEffect()->data()->modified_and_replaced_map();
  if (replace_map.find(target) != replace_map.end()) {
    // object is a temporary node of side effect result. Maybe escaped
    return true;
  }
  // NOTE: it's temporary solution before object reference graph completed
  return CheckReferenced(target->GetGraph(), target);
}

}  // namespace pijit
}  // namespace mindspore
