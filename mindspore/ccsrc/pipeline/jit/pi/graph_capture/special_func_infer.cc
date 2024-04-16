
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
#include "pipeline/jit/pi/graph_capture/special_func_infer.h"
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "pipeline/jit/pi/common.h"
#include "pipeline/jit/pi/external.h"
#include "pipeline/jit/pi/graph_capture/graph_build.h"
#include "pipeline/jit/pi/graph_guard/infer.h"
#include "pipeline/jit/pi/graph_capture/side_effect.h"

namespace mindspore {
namespace pijit {
extern ValueNode *GetBoundSelf(CallNode *call_node);
extern void LogGuardFailed(ValueNode *node, const GraphJitConfig &conf, const std::string &msg);
extern AObject *InferFuncResult(const py::object &func, const std::vector<AObject *> &stack_args, int opcode,
                                const GraphJitConfig &conf, bool clear_guard);
extern AObject *InferFuncResult(const py::object &func, const py::object &args, const py::object &kwargs,
                                const GraphJitConfig &conf, bool clear_guard);

constexpr const char *kModuleName = "mindspore._extends.pijit.pijit_func_white_list";
constexpr const char *kFuncMapName = "_func_map";
constexpr const char *kSlotCallName = "__call__";

static bool CheckConstexpr(const py::object &func);

template <AObject::Type type>
static bool SetCallResType(CallNode *call_node) {
  call_node->SetVobj(AObject::MakeAObject(type));
  call_node->SetSubGraph(nullptr);
  return false;
}

bool JustCallAndSetRes(CallNode *call_node) {
  py::object func = call_node->input(0)->GetVobj()->GetPyObject();
  if (func.ptr() == nullptr) {
    return SetCallResType<AObject::kTypeAnyValue>(call_node);
  }

  std::vector<py::object> args;
  std::transform(call_node->getInputs().begin() + 1, call_node->getInputs().end(), std::back_inserter(args),
                 [](ValueNode *n) { return n->GetVobj() ? n->GetVobj()->GetPyObject() : py::object(); });
  auto pair = Utils::PackCallStackArgs(args, call_node->GetOpcode());
  if (pair.first.ptr() == nullptr) {
    return SetCallResType<AObject::kTypeAnyValue>(call_node);
  }

  pi_jit_disable();
  PyObject *value = PyObject_Call(func.ptr(), pair.first.ptr(), pair.second.ptr());
  if (PyErr_Occurred()) {
    MS_LOG(ERROR) << "got an error " << py::error_already_set().what() << " at call the "
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
  call_node->SetSubGraph(sub_graph);
  ret_node->SetGraph(call_node->GetGraph());

  sub_graph->SetRetVal(ret_node);
  call_node->SetInlineReason(InlineReason::kInline);
  return true;
}

bool GuardConstCallNodeParam(CallNode *call_node, Graph *sub_graph, int max_guard_depth) {
  std::vector<std::pair<TracePtr, GuardLevel>> traces;
  for (auto i : call_node->getInputs()) {
    if (i->IsConstantValue()) {
      continue;
    }
    AObject::Type type = i->GetVobj() ? i->GetVobj()->GetType() : AObject::kTypeAnyValue;
    if (type == AObject::kTypeAnyValue) {
      return false;
    }
    TracePtr tr = sub_graph->TraceValueNode(i, max_guard_depth);
    if (tr == nullptr) {
      if (static_cast<size_t>(max_guard_depth) >= INT_MAX) {
        LogGuardFailed(i, sub_graph->Config(), "GuardConstCannNodeParm failed");
      }
      return false;
    }
    GuardLevel level = GuardLevel::GEqual;
    if (type == AObject::kTypeTensor) {
      if (i->GetOpcode() == LOAD_GLOBAL) {
        level = GuardLevel::GId;  // only guard global tensor
      } else {
        level = GuardLevel::GDeduce;
      }
    }
    traces.push_back({tr, level});
  }

  const auto &guard = sub_graph->GetGuard()->GetGuard();
  guard->Backup();
  for (const auto &i : traces) {
    if (!guard->GuardOn(i.first, i.second)) {
      guard->Rollback();
      return false;
    }
  }
  guard->Pop();
  return true;
}

static bool InferConvertMap(CallNode *call_node) {
  AObject *func_info = call_node->input(0)->GetVobj();
  func_info->SetMsFlag(AObject::kMsFlagStandardFunc);
  py::object func = func_info->GetPyObject();
  py::object tmp = Utils::GetModuleAttr("mindspore._extends.parse.resources", "convert_object_map");
  auto dict_obj = py::cast<py::dict>(tmp);
  auto infer_obj = dict_obj[func];
  AObject *res = nullptr;
  call_node->SetSubGraph(nullptr);
  SetCallResType<AObject::kTypeTensor>(call_node);
  if (PyFunction_Check(infer_obj.ptr())) {
    MS_LOG(DEBUG) << "infer function " << std::string(py::str(PyFunction_GET_CODE(infer_obj.ptr())));
    int op = call_node->GetOpcode();
    const auto &conf = call_node->GetGraph()->Config();
    std::vector<AObject *> args;
    std::transform(call_node->getInputs().begin() + 1, call_node->getInputs().end(), std::back_inserter(args),
                   [](ValueNode *n) { return n->GetVobj(); });
    res = InferFuncResult(func, {args.begin() + 1, args.end()}, op, conf, true);
  } else if (IsPrimitiveType<true>(Py_TYPE(infer_obj.ptr()))) {
    MS_LOG(DEBUG) << "infer primitive " << std::string(py::str(infer_obj));
    std::vector<PyObject *> list;
    bool infer_fail = false;
    for (size_t i = 1; !infer_fail && i < call_node->getInputs().size(); i++) {
      AObject *p = call_node->input(i)->GetVobj();
      PyObject *o = p ? p->GetPyObject().ptr() : nullptr;
      list.push_back(o);
      infer_fail = o == nullptr;
    }
    if (infer_fail) {
      return false;
    }
    auto inst = mindspore::pijit::InferEngine::GetInstance();
    bool is_abstract = false;
    PyObject *ret = inst->InferPrimitive(infer_obj.ptr(), list, &is_abstract);
    if (ret == nullptr) {
      return false;
    }
    AObject::Type type = AObject::GetPyType(ret);
    res = is_abstract && type != AObject::kTypeTensor ? AObject::MakeAObject(type) : AObject::Convert(ret);
    Py_DECREF(ret);
  } else {
    return false;
  }
  if (res) {
    call_node->SetVobj(res);
  }
  return false;
}

static bool InferGetCachePrim(CallNode *n) {
  // just return the first parameter of _get_cache_prim
  Graph *g = n->GetSubGraph();
  n->SetVobj(n->input(1)->GetVobj());
  g->SetRetVal(n->input(1));
  return true;
}

static bool InferRegistryGet(CallNode *call_node) {
  Graph *g = call_node->GetSubGraph();
  JustCallAndSetRes(call_node);

  py::object func = call_node->GetVobj()->GetPyObject();
  if (call_node->getInputs().back()->GetOpcode() == LOAD_CONST && func.ptr() != nullptr) {
    return CallNodeReturnConst(call_node, g, call_node->GetVobj());
  }
  return false;
}

static bool InferPrimitive(CallNode *call_node) {
  static const std::unordered_map<std::string, AObject::Type> not_ret_tensor_prim = {
    {"Prim[_get_grad_op]<constexpr_prim=True>", AObject::kTypeMetaFuncGraph},
    {"Prim[DType]", AObject::kTypeAnyValue},
    {"Prim[Partial]<side_effect_propagate=1>", AObject::kTypeAnyValue},
  };
  Graph *sub_graph = call_node->GetSubGraph();
  call_node->SetVobj(AObject::MakeAObject(AObject::kTypeTensor));
  call_node->SetSubGraph(nullptr);
  PyObject *prim = call_node->input(0)->GetVobj()->GetPyObject().ptr();
  std::string prim_key = std::string(py::str(prim));
  if (prim_key == "Prim[_get_grad_op]<constexpr_prim=True>") {
    py::object grad_class = Utils::GetModuleAttr("mindspore._c_expression", "GradOperation_");
    AbstractType *type = static_cast<AbstractType *>(AObject::Convert(grad_class));
    AObject *res = type != nullptr ? type->BuildAbstractInstance({}, CALL_FUNCTION)
                                   : AObject::MakeAObject(AObject::kTypeMetaFuncGraph);
    call_node->SetVobj(res);
    return false;
  }

  auto iter = not_ret_tensor_prim.find(prim_key);
  if (iter != not_ret_tensor_prim.end()) {
    call_node->SetVobj(AObject::MakeAObject(iter->second));
  } else {
    call_node->SetVobj(AObject::MakeAObject(AObject::kTypeTensor));
  }

  std::vector<PyObject *> list;
  bool infer_fail = false;
  for (size_t i = 1; !infer_fail && i < call_node->getInputs().size(); i++) {
    AObject *p = call_node->input(i)->GetVobj();
    if (p == nullptr) {
      infer_fail = true;
      break;
    }
    PyObject *o;
    if (p->GetType() == AObject::kTypeTensor) {
      o = static_cast<AbstractTensor *>(p)->GetTensor(true).ptr();
    } else {
      o = p->GetPyObject().ptr();
    }
    list.push_back(o);
    infer_fail = o == nullptr;
  }
  if (infer_fail) {
    return false;
  }

  auto inst = mindspore::pijit::InferEngine::GetInstance();
  bool is_abstract = false;
  PyObject *ret;
  try {
    ret = inst->InferPrimitive(prim, list, &is_abstract);
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "infer primitive failed. reason:";
    MS_LOG(ERROR) << e.what();
    ret = nullptr;
  }
  if (ret == nullptr) {
    return false;
  }

  AObject::Type type = AObject::GetPyType(ret);
  AObject *type_info = is_abstract && type != AObject::kTypeTensor ? AObject::MakeAObject(type) : AObject::Convert(ret);
  call_node->SetVobj(type_info);
  Py_DECREF(ret);

  ConstantInfo::CollectPrimitiveConstantInfo(call_node);
  if (call_node->IsConstantValue()) {
    return CallNodeReturnConst(call_node, sub_graph, call_node->GetVobj());
  }
  return false;
}

static bool InferGradOperation(CallNode *call_node, AObject::MindsporeFlag f) {
  call_node->SetSubGraph(nullptr);
  AObject *grad_func = AObject::MakeAObject(AObject::kTypeFunction);
  grad_func->SetMsFlag(f);
  call_node->SetVobj(grad_func);
  py::object func = GraphBuilder::FindPyFunc(call_node->input(1)->GetVobj());
  if (func.ptr() == nullptr) {
    return false;
  }
  (void)pi_jit_should_compile(func, py::dict());
  auto jcr = getJitCompileResults(PyFunction_GET_CODE(func.ptr()));
  *jcr->conf = call_node->GetGraph()->Config();
  return false;
}

static bool InferMetaFunc(CallNode *call_node) {
  call_node->SetSubGraph(nullptr);
  const auto &vo = call_node->input(0)->GetVobj();
  MS_EXCEPTION_IF_CHECK_FAIL(vo->GetType() != AObject::kTypeType, "class call is before ");
  PyTypeObject *tp = vo->GetTypeObject();
  if (IsGradOperationType<true>(tp)) {
    // set grad flag
    return InferGradOperation(call_node, AObject::MindsporeFlag::kMsFlagGradFunc);
  } else if (IsVmapOperationType<true>(tp)) {
    // set vmap flag
    return InferGradOperation(call_node, AObject::MindsporeFlag::kMsFlagVmapFunc);
  } else if (IsShardType<true>(tp)) {
    // set shard flag
    return InferGradOperation(call_node, AObject::MindsporeFlag::kMsFlagShardFunc);
  }
  return false;
}

/**
 * find first free variable in names from function
 */
static py::object FindClosure(const py::object &o, const std::vector<std::string> &names, TracePtr *trace, bool strict,
                              bool print) {
  PyObject *func = o.ptr();
  if (PyMethod_Check(func)) {
    func = PyMethod_GET_FUNCTION(func);
  }
  if (!PyFunction_Check(func)) {
    return py::object();
  }
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(func));
  PyObject *closure = PyFunction_GET_CLOSURE(func);
  Py_ssize_t i = PyTuple_GET_SIZE(co->co_freevars) - 1;
  bool find = false;
  for (; i >= 0 && !find; --i) {
    std::string name = PyUnicode_AsUTF8(PyTuple_GET_ITEM(co->co_freevars, i));
    find = std::find(names.begin(), names.end(), name) != names.end();
  }
  if (!find) {
    return py::object();
  }
  Py_ssize_t idx = i + 1;
  PyObject *cell = PyTuple_GET_ITEM(closure, idx);
  PyObject *content = PyCell_GET(cell);
  if (trace) {
    TracePtr attr = CreateOpTrace(closure, LOAD_ATTR, 0, {*trace}, "", "__closure__", strict, print);
    TracePtr cc = CreateOpTrace(cell, BINARY_SUBSCR, 0, {attr, std::make_shared<ConstTrace>(py::int_(idx).ptr(), -1)},
                                "", "", strict, print);
    *trace = CreateOpTrace(content, LOAD_ATTR, 0, {cc}, "", "cell_contents", strict, print);
  }
  return py::cast<py::object>(content);
}

/**
 * get decorated function from 'after_grad'
 * \param after_grad _Grad.__call__.<locals>.after_grad
 * \return decorated object
 */
static py::object GetGradDecorated(const py::object &after_grad, TracePtr *trace, bool strict, bool print) {
  MS_ASSERT(PyFunction_Check(after_grad.ptr()));
  py::object decorated = FindClosure(after_grad, {"fn", "fn_"}, trace, strict, print);
  MS_EXCEPTION_IF_CHECK_FAIL(decorated.ptr() != nullptr, "can't find decorated function 'fn' or 'fn_' from " +
                                                           std::string(py::str(after_grad.ptr())));
  if (!PyFunction_Check(decorated.ptr())) {
    return decorated;
  }
  std::string decorated_name = PyUnicode_AsUTF8(reinterpret_cast<PyFunctionObject *>(decorated.ptr())->func_qualname);
  if (decorated_name == "_Grad.__call__.<locals>.aux_fn") {
    decorated = FindClosure(decorated, {"fn"}, trace, strict, print);
    MS_EXCEPTION_IF_CHECK_FAIL(decorated.ptr() != nullptr, "can't find decorated function 'fn' from " + decorated_name);
  }
  return decorated;
}

static py::object DeleteGradSensArgs(const py::object &args, const py::object &kwargs) {
  // sens param specified in kwargs
  if (kwargs.ptr() != nullptr && PyDict_DelItemString(kwargs.ptr(), "sens_param") != -1) {
    return args;
  }
  PyErr_Clear();
  // sens param is the last position argument
  PyObject *new_arg = PyTuple_GetSlice(args.ptr(), 0, PyTuple_GET_SIZE(args.ptr()) - 1);
  return py::reinterpret_steal<py::object>(new_arg);
}

static AObject *InferGradFuncResult(const py::object &func, const py::object &args, const py::object &kwargs,
                                    const GraphJitConfig &conf) {
  auto jcr = getJitCompileResults(func.ptr());
  *jcr->conf = conf;
  return InferFuncResult(func, args, kwargs, conf, true);
}

/**
 * Use the function decorated by 'after_grad' and arguments of 'after_grad' when called to infer result.
 * If the function has no unsupported operation, merge the guard of inferred graph to caller graph.
 * else clear the mask of mindspore flag, avoid to capture this function call
 */
void HandleGradFuncCall(CallNode *call_node, AObject *decorated, bool sens_param) {
  const int except_flag = AObject::kMsFlagGradFunc | AObject::kMsFlagShardFunc | AObject::kMsFlagVmapFunc;
  ValueNode *grad_func_node = call_node->input(0);
  std::vector<py::object> stack_args;
  py::object func;
  py::object args;
  py::object kwargs;

  // prepare parameters
  bool param_ready = decorated->GetPyObject().ptr() != nullptr;
  for (size_t i = 1; param_ready && i < call_node->getInputs().size(); ++i) {
    AObject *tmp = call_node->input(i)->GetVobj();
    stack_args.emplace_back(tmp != nullptr ? tmp->GetPyObject() : py::object());
    param_ready = stack_args.back().ptr() != nullptr;
  }
  if (param_ready) {
    auto pair = Utils::PackCallStackArgs(stack_args, call_node->GetOpcode());
    args = pair.first;
    kwargs = pair.second;
    param_ready = pair.first.ptr() != nullptr;
  }
  if (!param_ready) {
    call_node->SetInlineReason(InlineReason::kInlineInfer_Fail);
    grad_func_node->GetVobj()->ClearMsFlag(except_flag);
    return;
  }
  if (sens_param) {
    args = DeleteGradSensArgs(args, kwargs);
  }

  // get callable
  if (decorated->GetType() != AObject::kTypeCell) {
    MS_EXCEPTION_IF_CHECK_FAIL(decorated->GetType() == AObject::kTypeFunction, "check grad input");
    func = decorated->GetPyObject();
  } else {
    // here get bound method.
    func = decorated->GetAttr(GraphBuilder::ID_construct)->GetPyObject();
  }

  AObject *res = InferGradFuncResult(func, args, kwargs, call_node->GetGraph()->Config());
  if (res == nullptr || !res->IsMindSporeSupportedType()) {
    call_node->SetInlineReason(InlineReason::kInlineInfer_Fail);
    grad_func_node->GetVobj()->ClearMsFlag(except_flag);
    return;
  }
  call_node->SetInlineReason(InlineReason::kInlineGraphSupportedByMS);
  call_node->SetVobj(res);
}

static void HandleGradFunc(CallNode *call_node, const py::object &after_grad, TracePtr *trace) {
  auto config = call_node->GetGraph()->Config();
  bool strict = config.GetBoolConfig(GraphJitConfig::kStrictTrace);
  bool print = config.GetBoolConfig(GraphJitConfig::kPrintGuard);
  py::object decorated_func = GetGradDecorated(after_grad, trace, strict, print);
  TracePtr ptr = *trace;
  py::object grad = FindClosure(after_grad, {"grad_", "self"}, &ptr, strict, print);
  MS_EXCEPTION_IF_CHECK_FAIL(grad.ptr() != nullptr,
                             "can't find 'grad_' object from " + std::string(py::str(after_grad.ptr())));
  bool sens_param = grad.attr("sens_param").ptr() == Py_True;
  MS_LOG(DEBUG) << "infer function 'after_grad', has sens_param " << (sens_param ? "True" : "False");

  auto guard = call_node->GetGraph()->GetGuard()->GetGuard();
  guard->GuardOn(*trace, mindspore::pijit::GuardLevel::GEqual);
  if (config.GetBoolConfig(GraphJitConfig::kGuardDetachObject)) {
    (*trace)->Detach();
  }
  call_node->SetSubGraph(nullptr);
  HandleGradFuncCall(call_node, AObject::Convert(decorated_func), sens_param);
}

static bool InferGradFunc(CallNode *call_node) {
  AObject *vo = call_node->input(0)->GetVobj();
  vo->SetMsFlag(AObject::kMsFlagGradFunc);
  py::object after_grad = vo->GetPyObject();
  TracePtr trace = call_node->GetGraph()->TraceValueNode(call_node->input(0));
  if (trace == nullptr) {
    vo->ClearMsFlag(AObject::kMsFlagGradFunc);
    call_node->SetSubGraph(nullptr);
    return false;
  }
  HandleGradFunc(call_node, after_grad, &trace);
  return false;
}

static bool InferMSConstexpr(CallNode *call_node) {
  Graph *g = call_node->GetSubGraph();
  JustCallAndSetRes(call_node);

  py::object cnst = call_node->GetVobj()->GetPyObject();
  if (cnst.ptr() == nullptr) {
    return false;
  }
  bool is_constexpr = CheckConstexpr(call_node->input(0)->GetVobj()->GetPyObject());
  if (is_constexpr || GuardConstCallNodeParam(call_node, g, 2)) {
    return CallNodeReturnConst(call_node, g, call_node->GetVobj());
  }
  return false;
}

static bool GuardBuiltinFunc(CallNode *call_node) {
  Graph *graph = call_node->GetGraph();
  for (auto i : call_node->getInputs()) {
    if (i->GetVobj() && i->GetVobj()->GetType() == AObject::kTypeTensor) {
      AbstractTensor *tensor = static_cast<AbstractTensor *>(i->GetVobj());
      if (!tensor->IsStubTensor() && !CheckTensorDataInitialized(tensor->GetPyObject())) {
        // fake value
        return false;
      }
    }
  }
  return graph->GuardValueNode(call_node);
}

static bool GuardIsInstance(CallNode *call_node) {
  Graph *graph = call_node->GetGraph();
  const auto &cnst = call_node->input(1)->GetConstantInfo();
  if (cnst != nullptr && cnst->type() != nullptr) {
    constexpr int second_arg = 2;
    return graph->GuardValueNode(call_node->input(second_arg));
  }
  return graph->GuardValueNode(call_node);
}

bool InferBuiltinFuncOrMethod(CallNode *call_node) {
  Graph *sub_graph = call_node->GetSubGraph();
  (void)JustCallAndSetRes(call_node);
  ConstantInfo::CollectBuiltinFuncConstantInfo(call_node);
  if (call_node->IsConstantValue()) {
    return CallNodeReturnConst(call_node, sub_graph, call_node->GetVobj());
  }
  if (call_node->GetVobj() == nullptr || call_node->GetVobj()->GetPyObject().ptr() == nullptr) {
    return false;
  }

  bool guard_success = false;
  std::string name = GetFuncName(call_node->input(0)->GetVobj()->GetPyObject());
  if (name == "isinstance") {
    guard_success = GuardIsInstance(call_node);
  } else {
    guard_success = GuardBuiltinFunc(call_node);
  }
  if (guard_success) {
    return CallNodeReturnConst(call_node, sub_graph, call_node->GetVobj());
  }
  return false;
}

static bool InferTensorAsType(CallNode *call_node) {
  ValueNode *self_node = GetBoundSelf(call_node);
  bool is_not_method = call_node->input(0)->GetVobj()->GetType() != AObject::kTypeBoundMethod;
  ValueNode *dtype_node = call_node->input(1 + is_not_method);

  Graph *sub_graph = call_node->GetSubGraph();

  py::object prim_cast = Utils::GetModuleAttr("mindspore.ops.functional", "cast", false, true);

  PyTypeObject *tp = Py_TYPE(prim_cast.ptr());
  std::stringstream s;
  s << (tp->tp_name ? tp->tp_name : "<unnamed>") << "<" << prim_cast.ptr() << ">";

  ValueNode *prim_node = sub_graph->NewValueNode(AObject::Convert(prim_cast), LOAD_CONST, -1, {});

  std::vector<ValueNode *> cast_args = {prim_node, self_node, dtype_node};
  CallNode *ret_node = sub_graph->NewCallNode(CALL_FUNCTION, cast_args.size() - 1, cast_args);
  ret_node->SetGraph(sub_graph);
  (void)InferPrimitive(ret_node);

  sub_graph->GetTracedNodes().push_back(prim_node);
  sub_graph->GetTracedNodes().push_back(ret_node);
  sub_graph->SetRetVal(ret_node);

  call_node->SetSubGraph(sub_graph);
  call_node->SetVobj(ret_node->GetVobj());
  call_node->SetInlineReason(InlineReason::kInline);
  return true;
}

bool InferListAppend(CallNode *call_node) {
  Graph *sub_graph = call_node->GetSubGraph();
  call_node->SetSubGraph(nullptr);

  ValueNode *method_node = call_node->input(0);
  if (method_node->GetOpcode() != LOAD_ATTR) {
    return false;
  }
  ValueNode *self = method_node->input(0);
  if (self->GetOpcode() != BUILD_LIST) {
    // guard old_list length, transform to "new_list = [old_list[0], old_list[1], ... , new_element]".
    return false;
  }
  std::vector<ValueNode *> inputs = self->getInputs();
  inputs.push_back(call_node->input(1));

  std::vector<AObject *> tmp;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(tmp), [](ValueNode *n) { return n->GetVobj(); });
  AObject *list_info = AObject::BuildOperations(tmp, BUILD_LIST);
  ValueNode *ret_node = sub_graph->NewValueNode(list_info, BUILD_LIST, inputs.size(), inputs);

  sub_graph->GetTracedNodes().push_back(ret_node);
  sub_graph->SetRetVal(ret_node);

  call_node->SetSubGraph(sub_graph);
  call_node->SetVobj(ret_node->GetVobj());
  call_node->SetInlineReason(InlineReason::kInline);
  return true;
}

static bool InferPopAsGet(CallNode *call_node) {
  py::object func = call_node->input(0)->GetVobj()->GetPyObject();
  PyObject *fPtr = func.ptr();
  if (func.ptr() == nullptr || PyMethod_Check(fPtr)) {
    return SetCallResType<AObject::kTypeAnyValue>(call_node);
  }

  std::vector<py::object> args;
  std::transform(call_node->getInputs().begin() + 1, call_node->getInputs().end(), std::back_inserter(args),
                 [](ValueNode *n) { return n->GetVobj() ? n->GetVobj()->GetPyObject() : py::object(); });
  auto pair = Utils::PackCallStackArgs(args, call_node->GetOpcode());
  if (pair.first.ptr() == nullptr) {
    return SetCallResType<AObject::kTypeAnyValue>(call_node);
  }
  PyObject *value = PyObject_Call(func.ptr(), pair.first.ptr(), pair.second.ptr());
  call_node->SetVobj(AObject::Convert(value));
  call_node->SetSubGraph(nullptr);
  call_node->GetGraph()->GetSideEffect()->SetSideEffectNode(call_node);
  return false;
}

static bool SetForbiddenFuncInfo(CallNode *call_node) {
  SetCallResType<AObject::kTypeAnyValue>(call_node);
  call_node->SetInlineReason(InlineReason::kInlineFunc_Type_Unsupported);
  return false;
}

bool InferMsApiFunc(CallNode *call_node) {
  Graph *sub_graph = call_node->GetSubGraph();
  SetCallResType<AObject::kTypeAnyValue>(call_node);
  if (call_node->input(0)->GetVobj() == nullptr || call_node->input(0)->GetVobj()->GetPyObject().ptr() == nullptr) {
    return false;
  }

  py::object callable_object = call_node->input(0)->GetVobj()->GetPyObject();
  std::vector<py::object> args;
  std::transform(call_node->getInputs().begin() + 1, call_node->getInputs().end(), std::back_inserter(args),
                 [](ValueNode *n) { return n->GetVobj() ? n->GetVobj()->GetPyObject() : py::object(); });
  auto pair = Utils::PackCallStackArgs(args, call_node->GetOpcode());
  if (pair.first.ptr() == nullptr) {
    return false;
  }
  PyTypeObject *callable_type = Py_TYPE(callable_object.ptr());

  AObject *info;

  bool enable_func_graph_eval = kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kEnableMsApiInfer);
  if (enable_func_graph_eval) {
    py::object res = EvalMSAPIValue(callable_object, pair.first, pair.second);
    info = AObject::Convert(res);
  } else if (IsPrimitiveType<true>(callable_type) || IsPrimitiveFunctionType<true>(callable_type)) {
    call_node->SetSubGraph(sub_graph);
    return InferPrimitive(call_node);
  } else {
    info = InferFuncResult(callable_object, pair.first, pair.second, call_node->GetGraph()->Config(), true);
  }

  call_node->SetVobj(info);
  if (info->GetPyObject().ptr() != nullptr) {
    ConstantInfo::CollectBuiltinFuncConstantInfo(call_node);
    call_node->input(0)->GetVobj()->SetMsFlag(AObject::kMsFlagStandardFunc);
  }
  if (call_node->IsConstantValue()) {
    return CallNodeReturnConst(call_node, sub_graph, call_node->GetVobj());
  }
  return false;
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
  FUNC_KEY_PSJIT_CODE,            // "mindspore.common.api.jit.<locals>.staging_specialize"
  FUNC_KEY_CONSTEXPR,             // "mindspore.ops.primitive.constexpr"
  FUNC_KEY_PRIMEXPR,              // "mindspore.ops.primitive._primexpr"
  FUNC_KEY_GET_CACHE_PRIM,        // "mindspore.ops._primitive_cache._get_cache_prim"
  FUNC_KEY_REGISTRY_GET,          // "mindspore.common._register_for_tensor.Registry.get"
  FUNC_KEY_TENSOR_ASTYPE,         // "mindspore.common.tensor.Tensor.astype"
  FUNC_KEY_GRAD_OPERATIONS_CODE,  // "mindspore.ops.composite.base._Grad.__call__.<locals>.after_grad"
  FUNC_KEY_PSJIT_CONVERTMAP,      // "mindspore._extends.parse.resources.convert_object_map"
  FUNC_KEY_GRAPH_CELL,            // "mindspore.nn.cell.GraphCell"
  FUNC_KEY_MS_API,                // mindspore api
  FUNC_KEY_COUNT,
};
static FuncKey FindFuncKey(const py::object &callable);

static const std::unordered_map<FuncKey, InferFunc> infer_func_map = {
  {FUNC_KEY_PIJIT_CONSTEXPR, JustCallAndSetRes},
  {FUNC_KEY_PIJIT_FORBIDDEN, SetForbiddenFuncInfo},
  {FUNC_KEY_BUILTIN_FUNC, InferBuiltinFuncOrMethod},
  {FUNC_KEY_LIST_APPEND, InferListAppend},
  {FUNC_KEY_DICT_POP, InferPopAsGet},
  {FUNC_KEY_PRIMITIVE, InferPrimitive},
  {FUNC_KEY_META_FUNCG_RAPH, InferMetaFunc},
  {FUNC_KEY_PSJIT_CODE, SetCallResType<AObject::kTypeTensor>},
  {FUNC_KEY_CONSTEXPR, InferMSConstexpr},
  {FUNC_KEY_PRIMEXPR, InferMSConstexpr},
  {FUNC_KEY_GET_CACHE_PRIM, InferGetCachePrim},
  {FUNC_KEY_REGISTRY_GET, InferRegistryGet},
  {FUNC_KEY_TENSOR_ASTYPE, InferTensorAsType},
  {FUNC_KEY_GRAD_OPERATIONS_CODE, InferGradFunc},
  {FUNC_KEY_PSJIT_CONVERTMAP, InferConvertMap},
  {FUNC_KEY_GRAPH_CELL, SetCallResType<AObject::kTypeTensor>},
  {FUNC_KEY_MS_API, InferMsApiFunc},
};

static const std::unordered_map<FuncKey, InferFunc> mind_infer_func_map = {
  {FUNC_KEY_PIJIT_CONSTEXPR, JustCallAndSetRes},     {FUNC_KEY_PIJIT_FORBIDDEN, SetForbiddenFuncInfo},
  {FUNC_KEY_BUILTIN_FUNC, InferBuiltinFuncOrMethod}, {FUNC_KEY_PSJIT_CODE, SetCallResType<AObject::kTypeTensor>},
  {FUNC_KEY_GET_CACHE_PRIM, InferGetCachePrim},      {FUNC_KEY_REGISTRY_GET, InferRegistryGet},
};

InferFunc FindInferFunc(const py::object &callable, bool trace_flag) {
  FuncKey k = FindFuncKey(callable);
  const auto &map = trace_flag ? mind_infer_func_map : infer_func_map;
  auto iter = map.find(k);
  if (iter != map.end()) {
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
  MS_EXCEPTION_IF_CHECK_FAIL(PyDict_CheckExact(func_map.ptr()), "white list func map must be 'dict[int, str]'");
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

static FuncKey KeyFinderMetaFunc(const py::object &callable) {
  PyTypeObject *type_object = reinterpret_cast<PyTypeObject *>(callable.ptr());
  type_object = PyType_CheckExact(type_object) ? type_object : Py_TYPE(type_object);
  return IsMetaFuncGraphType<true>(type_object) ? FUNC_KEY_META_FUNCG_RAPH : FUNC_KEY_EMPTY;
}

static FuncKey KeyFinderGraphCell(const py::object &callable) {
  static size_t id = 0;
  if (id == 0) {
    py::object type = Utils::GetModuleAttr("mindspore.nn.cell", "GraphCell", false, true);
    id = reinterpret_cast<size_t>(type.ptr());
  }
  PyTypeObject *type_object = reinterpret_cast<PyTypeObject *>(callable.ptr());
  type_object = PyType_CheckExact(type_object) ? type_object : Py_TYPE(type_object);
  size_t cur_id = reinterpret_cast<size_t>(type_object);
  return cur_id == id ? FUNC_KEY_GRAPH_CELL : FUNC_KEY_EMPTY;
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
  MS_LOG(DEBUG) << "func " << std::string(py::str(func_info)) << " is forbidden to analyze, module is " << mod;
  return FUNC_KEY_PIJIT_FORBIDDEN;
}

static FuncKey FindFuncKey(const py::object &callable) {
  if (callable.ptr() == nullptr || !PyCallable_Check(callable.ptr())) {
    return FUNC_KEY_EMPTY;
  }
  std::vector<FuncKey (*)(const py::object &callable)> finders = {
    KeyFinderFuncId,   KeyFinderFuncCodeId, KeyFinderPrimitive,
    KeyFinderMetaFunc, KeyFinderGraphCell,  KeyFinderSkipModule,  // must be last for check modules
  };
  FuncKey res = FUNC_KEY_EMPTY;
  for (auto iter = finders.begin(), end = finders.end(); iter != end && res == FUNC_KEY_EMPTY; ++iter) {
    res = (*iter)(callable);
  }
  return res;
}

bool CheckJitConstexpr(const py::object &func) {
  if (func.ptr() == nullptr) {
    return false;
  }
  FuncKey k = KeyFinderFuncId(func);
  return k == FUNC_KEY_PIJIT_CONSTEXPR;
}

static bool CheckConstexpr(const py::object &func) { return KeyFinderPrimitive(func) == FUNC_KEY_CONSTEXPR; }

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

}  // namespace pijit
}  // namespace mindspore
