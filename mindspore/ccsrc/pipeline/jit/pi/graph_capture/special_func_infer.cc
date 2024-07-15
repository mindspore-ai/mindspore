
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
#include "pipeline/jit/pi/runtime.h"
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
constexpr const size_t kDictPopParamsNum = 2;
constexpr const size_t BoundMethodInputSize = 2;

static bool CheckConstexpr(const py::object &func);

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
  std::transform(call_node->getInputs().begin() + 1, call_node->getInputs().end(), std::back_inserter(args),
                 [](ValueNode *n) { return n->GetVobj() ? n->GetVobj()->GetPyObject() : py::object(); });
  auto pair = Utils::PackCallStackArgs(args, call_node->GetOpcode());
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

static bool InferConvertMap(CallNode *call_node, GraphBuilder *unused = nullptr) {
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
  if (call_node->getInputs().back()->GetOpcode() == LOAD_CONST && func.ptr() != nullptr) {
    return CallNodeReturnConst(call_node, g, call_node->GetVobj());
  }
  return false;
}

static bool InferPrimitive(CallNode *call_node, GraphBuilder *unused = nullptr) {
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
  (void)pi_jit_should_compile(func, py::dict(), py::none());
  auto jcr = GetJitCompileResults(PyFunction_GET_CODE(func.ptr()));
  jcr->set_conf(std::make_shared<GraphJitConfig>(call_node->GetGraph()->Config()));
  return false;
}

static bool InferMetaFunc(CallNode *call_node, GraphBuilder *unused = nullptr) {
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

/**
 * Use the function decorated by 'after_grad' and arguments of 'after_grad' when called to infer result.
 * If the function has no unsupported operation, merge the guard of inferred graph to caller graph.
 * else clear the mask of mindspore flag, avoid to capture this function call
 */
void HandleGradFuncCall(CallNode *call_node, AObject *decorated, bool sens_param, const py::object &after_grad) {
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

  AObject *res = InferFuncResult(func, args, kwargs, call_node->GetGraph()->Config(), true);
  if (res == nullptr || !res->IsMindSporeSupportedType()) {
    call_node->SetInlineReason(InlineReason::kInlineInfer_Fail);
    grad_func_node->GetVobj()->ClearMsFlag(except_flag);
    return;
  }
  py::object infer_after_grad = Utils::GetModuleAttr(kModuleName, "infer_after_grad", true, true);
  py::object result;
  try {
    result = infer_after_grad(after_grad, args, res->GetPyObject());
  } catch (std::exception &e) {
    MS_LOG(WARNING) << "Error while infer_after_grad, error:" << e.what();
    PyErr_Clear();
  }
  if (result.ptr() != nullptr && result.ptr() != Py_None) {
    call_node->SetVobj(AObject::Convert(result));
  } else {
    call_node->SetVobj(res);
  }
  call_node->SetInlineReason(InlineReason::kInlineGraphSupportedByMS);
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
  HandleGradFuncCall(call_node, AObject::Convert(decorated_func), sens_param, after_grad);
}

static bool InferGradFunc(CallNode *call_node, GraphBuilder *unused = nullptr) {
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

static bool InferMSConstexpr(CallNode *call_node, GraphBuilder *unused = nullptr) {
  Graph *g = call_node->GetSubGraph();
  JustCallAndSetRes(call_node);

  py::object cnst = call_node->GetVobj()->GetPyObject();
  if (cnst.ptr() == nullptr) {
    return false;
  }
  bool is_constexpr = CheckConstexpr(call_node->input(0)->GetVobj()->GetPyObject());
  constexpr int max_guard_depth = 2;
  if (is_constexpr || GuardConstCallNodeParam(call_node, g, max_guard_depth)) {
    return CallNodeReturnConst(call_node, g, call_node->GetVobj());
  }
  return false;
}

static bool GuardBuiltinFunc(CallNode *call_node) {
  if (call_node->input(0)->GetVobj() == nullptr) {
    return false;
  }
  PyObject *func = call_node->input(0)->GetVobj()->GetPyObject().ptr();
  if (PyMethod_Check(func)) {
    auto self = PyMethod_GET_SELF(func);
    if (IsTensorType<true>(Py_TYPE(self)) && !CheckTensorDataInitialized(py::cast<py::object>(self))) {
      // fake value
      return false;
    }
  }
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

static bool InferTensorAsType(CallNode *call_node, GraphBuilder *unused = nullptr) {
  ValueNode *self_node = GetBoundSelf(call_node);
  bool is_not_method = call_node->input(0)->GetVobj()->GetType() != AObject::kTypeBoundMethod;
  ValueNode *dtype_node = call_node->input(1 + is_not_method);

  Graph *sub_graph = call_node->GetSubGraph();

  py::object prim_cast = Utils::GetModuleAttr("mindspore.ops.functional", "cast", false, true);

  PyTypeObject *tp = Py_TYPE(prim_cast.ptr());
  std::stringstream s;
  s << (tp->tp_name ? tp->tp_name : "<unnamed>") << "<" << prim_cast.ptr() << ">";

  ValueNode *prim_node = sub_graph->NewValueNode(AObject::Convert(prim_cast), LOAD_CONST, -1, {});

  if (dtype_node->GetVobj()->GetType() == AObject::kTypeString &&
      dtype_node->GetVobj()->GetPyObject().ptr() != nullptr) {
    auto dtypeStr = py::cast<std::string>(dtype_node->GetVobj()->GetPyObject());
    std::vector<std::string> under_line_dtype = {"bool", "int", "float", "list", "tuple"};
    if (std::find(under_line_dtype.begin(), under_line_dtype.end(), dtypeStr) != under_line_dtype.end()) {
      dtypeStr = dtypeStr + "_";
    }
    auto dtype_obj = Utils::GetModuleAttr("mindspore.common.dtype", dtypeStr, false, true);
    if (dtype_obj.ptr() != nullptr) {
      dtype_node = sub_graph->NewValueNode(AObject::Convert(dtype_obj), LOAD_CONST, -1, {});
    }
  }

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

static void RecordSideEffectCallNode(Graph *graph, CallNode *call_node, SideEffect::Type type, bool trace_flag) {
  const auto &side_effect = graph->GetSideEffect();
  ValueNode *side_effect_node;
  if (trace_flag) {
    side_effect_node = call_node;
  } else {
    side_effect_node = graph->NewCallNode(call_node->GetOpcode(), call_node->GetOparg(), call_node->getInputs());
    side_effect_node->SetVobj(AObject::MakeAObject(AObject::kTypeAnyValue));
    graph->GetTracedNodes().push_back(side_effect_node);
  }
  side_effect->Record(side_effect_node, type);
}

static bool InferListAppend(CallNode *call_node, GraphBuilder *parent) {
  call_node->SetSubGraph(nullptr);

  // check is supported type and get arguments
  bool is_method_descriptor = false;
  ValueNode *self = GetSelfFromListAppendCall(call_node, &is_method_descriptor);
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

  // constant fold and set node info
  auto builder = GraphBuilder::Creator(parent->root(), parent, nullptr, nullptr, parent->trace_flag());
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
  bool is_new_var = self->GetOpcode() == BUILD_LIST && replace_map.find(self) == replace_map.end();
  if (!is_new_var || is_referenced || self == new_element) {
    parent->GetGraph()->GetSideEffect()->data()->RecordModifiedAndReplacedNode(old_node, new_node);
    RecordSideEffectCallNode(parent->GetGraph(), call_node, SideEffect::kListAppend, parent->trace_flag());
  }
  return true;
}

static bool InferDictPop(CallNode *call_node, GraphBuilder *parent) {
  call_node->SetSubGraph(nullptr);

  bool is_method_descriptor = false;
  ValueNode *self = GetSelfFromListAppendCall(call_node, &is_method_descriptor);
  if (self == nullptr) {
    return false;
  }
  // guard dict key and convert to constant key map
  if (!parent->GetGraph()->GuardValueNode(self)) {
    return false;
  }

  ValueNode *dict_node = self;
  ValueNode *key_node = call_node->input(1 + is_method_descriptor);
  ValueNode *default_node = call_node->getInputs().size() > (kDictPopParamsNum + is_method_descriptor)
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
    return false;
  }

  // constant fold and set node info
  auto builder = GraphBuilder::Creator(parent->root(), parent, nullptr, nullptr, parent->trace_flag());
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
  bool is_new_var = self->GetOpcode() == BUILD_MAP && replace_map.find(self) == replace_map.end();
  if (!is_new_var || is_referenced) {
    parent->GetGraph()->GetSideEffect()->data()->RecordModifiedAndReplacedNode(old_node, new_node);
    RecordSideEffectCallNode(parent->GetGraph(), call_node, SideEffect::kDictPop, parent->trace_flag());
  }
  return true;
}

static bool SetForbiddenFuncInfo(CallNode *call_node, GraphBuilder *unused = nullptr) {
  SetCallResType<AObject::kTypeAnyValue>(call_node);
  call_node->SetInlineReason(InlineReason::kInlineFunc_Type_Unsupported);
  return false;
}

template <bool force_ms_api>
bool InferMsApiFunc(CallNode *call_node, GraphBuilder *unused = nullptr) {
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

  bool enable_func_graph_eval = force_ms_api || kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kEnableMsApiInfer);
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
    call_node->input(0)->GetVobj()->SetMsFlag(AObject::kMsFlagStandardFunc);
  }
  if (call_node->IsConstantValue()) {
    return CallNodeReturnConst(call_node, sub_graph, call_node->GetVobj());
  }
  return false;
}

bool InferMappingGet(CallNode *call_node, GraphBuilder *unused = nullptr) {
  if (call_node->getInputs().size() == BoundMethodInputSize &&
      call_node->input(0)->GetVobj()->GetType() == AbstractObjectBase::kTypeBoundMethod) {
    auto func_node = call_node->input(0);
    auto self = func_node->input(0);
    auto param_node = call_node->input(1);
    if (self->IsConstantValue() && param_node->IsConstantValue()) {
      Graph *g = call_node->GetSubGraph();
      JustCallAndSetRes(call_node);
      return CallNodeReturnConst(call_node, g, call_node->GetVobj());
    }
  }
  SetCallResType<AObject::kTypeAnyValue>(call_node);
  call_node->SetInlineReason(InlineReason::kInlineFunc_Type_Unsupported);
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
  FUNC_KEY_MAPPING_GET,           // mapping get
  FUNC_KEY_COUNT,
};
static FuncKey FindFuncKey(const py::object &callable);

static const std::unordered_map<FuncKey, InferFunc> infer_func_map = {
  {FUNC_KEY_PIJIT_CONSTEXPR, JustCallAndSetRes},
  {FUNC_KEY_PIJIT_FORBIDDEN, SetForbiddenFuncInfo},
  {FUNC_KEY_BUILTIN_FUNC, InferBuiltinFuncOrMethod},
  {FUNC_KEY_LIST_APPEND, InferListAppend},
  {FUNC_KEY_DICT_POP, InferDictPop},
  {FUNC_KEY_PRIMITIVE, InferPrimitive},
  {FUNC_KEY_META_FUNCG_RAPH, InferMetaFunc},
  {FUNC_KEY_PSJIT_CODE, InferMsApiFunc<true>},
  {FUNC_KEY_CONSTEXPR, InferMSConstexpr},
  {FUNC_KEY_PRIMEXPR, InferMSConstexpr},
  {FUNC_KEY_GET_CACHE_PRIM, InferGetCachePrim},
  {FUNC_KEY_REGISTRY_GET, InferRegistryGet},
  {FUNC_KEY_TENSOR_ASTYPE, InferTensorAsType},
  {FUNC_KEY_GRAD_OPERATIONS_CODE, InferGradFunc},
  {FUNC_KEY_PSJIT_CONVERTMAP, InferConvertMap},
  {FUNC_KEY_GRAPH_CELL, SetCallResType<AObject::kTypeTensor>},
  {FUNC_KEY_MS_API, InferMsApiFunc<false>},
  {FUNC_KEY_MAPPING_GET, InferMappingGet},
};

static const std::unordered_map<FuncKey, InferFunc> mind_infer_func_map = {
  {FUNC_KEY_PIJIT_CONSTEXPR, JustCallAndSetRes},     {FUNC_KEY_PIJIT_FORBIDDEN, SetForbiddenFuncInfo},
  {FUNC_KEY_LIST_APPEND, InferListAppend},           {FUNC_KEY_DICT_POP, InferDictPop},
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
  MS_LOG(DEBUG) << "func " << std::string(py::str(func_info)) << " is forbidden to analyze, module is " << mod;
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
