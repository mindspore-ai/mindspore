
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

// ------------------------------builtins functions--------------------------------
static constexpr const char *kBuiltinNameFunctionOrMethod = "builtin_function_or_method";
static constexpr const char *kBuiltinNameIsinstance = "isinstance";  // call __instancecheck__
static constexpr const char *kBuiltinNameIssubclass = "issubclass";  // call __subclasscheck__
static constexpr const char *kBuiltinNameLen = "len";                // call __len__
static constexpr const char *kBuiltinNameAbs = "abs";                // call __abs__
static constexpr const char *kBuiltinNameMax = "max";                // call __max__
static constexpr const char *kBuiltinNameLog = "log";                // call math.log
static constexpr const char *kBuiltinNameAll = "all";                // for each value in the iterable. call __bool__
static constexpr const char *kBuiltinNameAny = "any";                // for each value in the iterable. call __bool__
static constexpr const char *kBuiltinNameHash = "hash";              // call __hash__
static constexpr const char *kBuiltinNameId = "id";                  // no side effects
static constexpr const char *kBuiltinNameOrd = "ord";                // convert char to int. no side effect
static constexpr const char *kBuiltinNameCallable = "callable";      // no side effects
static constexpr const char *kBuiltinNameGetattr = "getattr";        // call __getattr__, or __getattribute__
static constexpr const char *kBuiltinNameHasattr = "hasattr";        // call __getattr__, or __getattribute__
// ------------------------------builtins functions--------------------------------

// ------------------------------builtins method--------------------------------
// static constexpr const char *kBuiltinNameUpdate = "update";  // dict update
static constexpr const char *kBuiltinNameAppend = "append";  // list update
// ------------------------------builtins method--------------------------------

// ------------------------------mindspore functions-------------------------------
static constexpr const char *kMindsporeNameGetCachePrim = "_get_cache_prim";
static constexpr const char *kMindsporeNameRegistryGet = "get";
static constexpr const char *kMindsporeNamePrimexpr = "CompileOp";
static constexpr const char *kMindsporeNameConstexpr = "ProxyOp";
/**
 * NOTE: mindspore/ops/composite/base.py, after_grad decorated by '_warp_func'
 * code name is 'wrapper', not 'after_grad', it only called by pynative
 */
static constexpr const char *kMindsporeNameGradFunc = "after_grad";
static constexpr const char *kMindsporeNameJitFunc = "staging_specialize";  // mindspore.jit
static constexpr const char *kMindsporeNamePrimitive = "Primitive_";
static constexpr const char *kMindsporeNameMetaFuncGraph = "MetaFuncGraph_";
static constexpr const char *kMindsporeNameTensorAsType = "astype";
static constexpr const char *kMindsporeNameMsCell = "mindspore.nn.Cell";
/**
 * convert function map
 * refer to convert_object_map in mindspore._extends.parse.resources.py
 */
static constexpr const char *kMindsporeNameConvertMap = "mindspore._extends.parse.resources.convert_object_map";
static constexpr const char *kMindsporeNameTensorInitCheck = "_init_check";
static constexpr const char *kMindsporeNameTensorContiguous = "contiguous";
// ------------------------------mindspore functions-------------------------------

static constexpr const char *kJitForbidden = ".pijit_forbidden";
static constexpr const char *kJitConstexpr = ".pijit_constexpr";

static const std::vector<std::string> tensor_module = {"mindspore.common.tensor", "mindtorch.torch.tensor"};
static const std::vector<std::string> bypass_function_whilelist = {kMindsporeNameTensorInitCheck,
                                                                   kMindsporeNameTensorContiguous};

static py::object GetGradClass() { return Utils::GetModuleAttr("mindspore._c_expression", "GradOperation_"); }

const char *GetFuncName(const py::object &f) {
  PyObject *func = f.ptr();
  if (PyMethod_Check(func)) {
    func = PyMethod_GET_FUNCTION(func);
  }
  if (PyCFunction_Check(func)) {
    return reinterpret_cast<PyCFunctionObject *>(func)->m_ml->ml_name;
  }
  PyCodeObject *co = nullptr;
  if (PyFunction_Check(func)) {
    co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(func));
  }
  if (co) {
    return PyUnicode_AsUTF8(co->co_name);
  }
  PyTypeObject *tp = PyType_Check(func) ? reinterpret_cast<PyTypeObject *>(func) : Py_TYPE(func);
  const char *res = strrchr(tp->tp_name, '.');
  return res ? res + 1 : tp->tp_name;
}

template <AObject::Type type>
bool SetCallResType(CallNode *call_node) {
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

  PyObject *value = PyObject_Call(func.ptr(), pair.first.ptr(), pair.second.ptr());
  if (PyErr_Occurred()) {
    MS_LOG(ERROR) << "got an error " << py::error_already_set().what() << " at call the "
                  << std::string(py::str(func.ptr()));
    PyErr_Clear();
  }
  call_node->SetVobj(AObject::Convert(value));
  call_node->SetSubGraph(nullptr);
  Py_XDECREF(value);
  return false;
}

bool CallNodeReturnConst(CallNode *call_node, Graph *sub_graph, AObject *value) {
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

static bool CheckConvertMap(const py::object &func) {
  if (func.ptr() == nullptr || !PyFunction_Check(func.ptr())) {
    return false;
  }
  py::object tmp = Utils::GetModuleAttr("mindspore._extends.parse.resources", "convert_object_map");
  auto dict_obj = py::cast<py::dict>(tmp);
  if (dict_obj.contains(func)) {
    return true;
  } else {
    return false;
  }
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

bool CheckGetCachePrim_(const py::object &f) {
  if (!PyFunction_Check(f.ptr())) {
    return false;
  }
  auto func_ptr = reinterpret_cast<PyFunctionObject *>(f.ptr());
  std::string name = PyUnicode_AsUTF8(func_ptr->func_module);
  bool is_func = name == "mindspore.ops._primitive_cache";
  return is_func;
}

bool InferGetCachePrim_(CallNode *n) {
  // just return the first parameter of _get_cache_prim
  Graph *g = n->GetSubGraph();
  n->SetVobj(n->input(1)->GetVobj());
  g->SetRetVal(n->input(1));
  return true;
}

bool IsTensorModule(const std::string &name) {
  return std::any_of(tensor_module.begin(), tensor_module.end(), [name](const auto &item) { return item == name; });
}

bool IsFuncInByPassWhiteList(const std::string &name) {
  return std::any_of(bypass_function_whilelist.begin(), bypass_function_whilelist.end(),
                     [name](const auto &item) { return item == name; });
}

bool CheckTensorBypass(const py::object &f) {
  if (!PyMethod_Check(f.ptr())) {
    return false;
  }
  auto func_ptr = reinterpret_cast<PyFunctionObject *>(PyMethod_Function(f.ptr()));
  std::string module = PyUnicode_AsUTF8(func_ptr->func_module);
  if (IsTensorModule(module)) {
    std::string func_name = GetFuncName(f);
    return IsFuncInByPassWhiteList(func_name);
  }
  return false;
}

bool InferTensorBypass(CallNode *n) {
  if (n->input(0)->GetOpcode() != LOAD_ATTR) {
    n->SetSubGraph(nullptr);
    return false;
  }
  Graph *g = n->GetSubGraph();
  n->SetVobj(AObject::Convert(PyMethod_Self(n->input(0)->GetVobj()->GetPyObject().ptr())));
  g->SetRetVal(n->input(0)->input(0));
  return true;
}

static bool CheckRegistryGet(const py::object &func) {
  PyObject *f = func.ptr();
  if (PyMethod_Check(f)) {
    f = PyMethod_GET_FUNCTION(f);
  }
  if (!PyFunction_Check(f)) {
    return false;
  }
  std::string name = PyUnicode_AsUTF8(reinterpret_cast<PyFunctionObject *>(f)->func_module);
  bool is_tensor = name == "mindspore.common._register_for_tensor";
  return is_tensor;
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

bool CheckPrimitive(const py::object &func) { return AObject::GetPyType(func.ptr()) == AObject::kTypePrimitive; }

bool InferPrimitive(CallNode *call_node) {
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
    AbstractType *type = static_cast<AbstractType *>(AObject::Convert(GetGradClass()));
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

bool InferGradOperation(CallNode *call_node, AObject::MindsporeFlag f) {
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

static bool CheckMetaFunc_(const py::object &o) {
  PyTypeObject *tp = PyType_Check(o.ptr()) ? reinterpret_cast<PyTypeObject *>(o.ptr()) : Py_TYPE(o.ptr());
  return IsMetaFuncGraphType<true>(tp);
}

static bool InferMetaFunc_(CallNode *call_node) {
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

static bool CheckGradFunc(const py::object &f) {
  if (!PyFunction_Check(f.ptr())) {
    return false;
  }
  std::string decorated_name = PyUnicode_AsUTF8(reinterpret_cast<PyFunctionObject *>(f.ptr())->func_qualname);
  return decorated_name == "_Grad.__call__.<locals>.after_grad" ||
         decorated_name == "GradOperation.__call__.<locals>.after_grad";
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

static bool CheckJitFunc(const py::object &o) {
  static const char except_file[] = "mindspore/common/api.py";
  static const size_t except_size = sizeof(except_file) - 1;
  PyObject *func = o.ptr();
  if (PyMethod_Check(func)) {
    func = PyMethod_GET_FUNCTION(func);
  }
  if (!PyFunction_Check(func)) {
    return false;
  }
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(func));
  const char *file = PyUnicode_AsUTF8(co->co_filename);
  const size_t size = strlen(file);
  return size > except_size && !strncmp(file + (size - except_size), except_file, except_size);
}

static bool CheckCell(const py::object &callable_info) {
  PyTypeObject *cell_type = PyType_Check(callable_info.ptr()) ? reinterpret_cast<PyTypeObject *>(callable_info.ptr())
                                                              : Py_TYPE(callable_info.ptr());
  if (!IsCellType<true>(cell_type)) {
    return false;
  }
  py::object tp = py::cast<py::object>(reinterpret_cast<PyObject *>(cell_type));
  std::string type_str = py::str(tp.ptr());
  const auto &sets = *kPIJitConfigDefault.getSetConfig(GraphJitConfig::kPSJitStrictCells);
  if (sets.find(type_str) != sets.end()) {
    return true;
  }

  // mindspore cells
  std::string m = tp.attr("__module__").cast<std::string>();
  constexpr const char except1[] = "mindspore.";
  constexpr int except1_size = sizeof(except1) - 1;
  if (!m.compare(0, except1_size, except1)) {
    kPIJitConfigDefault.AddPSJitStrictCells(type_str);
    return true;
  }
  return false;
}

static bool InferCell(CallNode *call_node) {
  PyTypeObject *cell_type = call_node->input(0)->GetVobj()->GetTypeObject();
  py::object tp = py::cast<py::object>(reinterpret_cast<PyObject *>(cell_type));

  const auto &conf = call_node->GetGraph()->Config();
  py::object func = tp.attr("construct");

  std::vector<AObject *> args;
  std::transform(call_node->getInputs().begin(), call_node->getInputs().end(), std::back_inserter(args),
                 [](ValueNode *n) { return n->GetVobj(); });
  AObject *res = InferFuncResult(func, args, call_node->GetOpcode(), conf, true);
  if (res == nullptr || res->GetType() == AObject::kTypeAnyValue) {
    res = AObject::MakeAObject(AObject::kTypeTensor);
  }

  call_node->SetVobj(res);
  call_node->SetSubGraph(nullptr);
  return false;
}

static bool CheckJitForbidden(const py::object &func) {
  if (func.ptr() == nullptr || PyCFunction_Check(func.ptr())) {
    return false;
  }
  std::string m = GetTopModule(func);
  const auto &l = *kPIJitConfigDefault.getSetConfig(GraphJitConfig::kAllowedInlineModules);
  bool allow_inline = l.find(m) != l.end();
  bool forbidden = !allow_inline || kPIJitConfigDefault.CheckJitForbidden(func);

  PyObject *func_info = func.ptr();
  if (PyMethod_Check(func_info)) {
    func_info = PyMethod_GET_FUNCTION(func_info);
  }
  if (!PyFunction_Check(func_info) && !PyCFunction_Check(func_info) && !PyType_Check(func_info)) {
    func_info = reinterpret_cast<PyObject *>(Py_TYPE(func_info));
  }
  MS_LOG(DEBUG) << "func " << std::string(py::str(func_info)) << (forbidden ? " is forbidden to" : " will ")
                << " Analyze, module is " << m;
  return forbidden;
}

bool CheckJitConstexpr(const py::object &func) {
  PyObject *op = func.ptr();
  if (op == nullptr) {
    return false;
  }
  if (PyMethod_Check(op)) {
    op = PyMethod_GET_FUNCTION(op);
  }
  return kPIJitConfigDefault.CheckJitConstexpr(py::cast<py::object>(op));
}

bool CheckMSConstexpr(const py::object &func) {
  std::string tp_name = py::str(reinterpret_cast<PyObject *>(Py_TYPE(func.ptr())));
  constexpr const char name[] = ".<locals>.decorator.<locals>.ProxyOp'>";
  constexpr const int size = sizeof(name) - 1;
  return tp_name.size() > size ? !tp_name.compare(tp_name.size() - size, size, name) : false;
}

bool CheckMSPrimexpr(const py::object &func) {
  std::string tp_name = py::str(reinterpret_cast<PyObject *>(Py_TYPE(func.ptr())));
  constexpr const char name[] = ".<locals>.deco.<locals>.CompileOp'>";
  constexpr const int size = sizeof(name) - 1;
  return tp_name.size() > size ? !tp_name.compare(tp_name.size() - size, size, name) : false;
}

static bool InferMSConstexpr(CallNode *call_node) {
  Graph *g = call_node->GetSubGraph();
  JustCallAndSetRes(call_node);

  py::object cnst = call_node->GetVobj()->GetPyObject();
  if (cnst.ptr() == nullptr) {
    return false;
  }
  if (!GuardConstCallNodeParam(call_node, g, 2)) {
    return false;
  }
  if (!CheckConstPyObject(cnst.ptr())) {
    MS_LOG(DEBUG) << std::string(py::str(cnst.ptr())) << " as const is unsupported";
    return false;
  }

  return CallNodeReturnConst(call_node, g, call_node->GetVobj());
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

#define DECLARE_BUILTIN_CFUNCTION(func_name)                 \
  p = PyDict_GetItemString(PyEval_GetBuiltins(), func_name); \
  MS_ASSERT(p &&PyCFunction_Check(p));                       \
  c_function_obj = PyCFunction_GET_FUNCTION(p);              \
  kBuiltinFuncOrMethodWhileList.emplace(c_function_obj);

static const std::set<PyCFunction> &GenCFunctionMap() {
  static std::set<PyCFunction> kBuiltinFuncOrMethodWhileList = {};
  if (!kBuiltinFuncOrMethodWhileList.empty()) {
    return kBuiltinFuncOrMethodWhileList;
  }
  PyCFunction c_function_obj = nullptr;
  PyObject *p = nullptr;
  DECLARE_BUILTIN_CFUNCTION(kBuiltinNameIsinstance);
  DECLARE_BUILTIN_CFUNCTION(kBuiltinNameIssubclass);
  DECLARE_BUILTIN_CFUNCTION(kBuiltinNameLen);
  DECLARE_BUILTIN_CFUNCTION(kBuiltinNameAbs);
  DECLARE_BUILTIN_CFUNCTION(kBuiltinNameMax);
  DECLARE_BUILTIN_CFUNCTION(kBuiltinNameAll);
  DECLARE_BUILTIN_CFUNCTION(kBuiltinNameAny);
  DECLARE_BUILTIN_CFUNCTION(kBuiltinNameHash);
  DECLARE_BUILTIN_CFUNCTION(kBuiltinNameId);
  DECLARE_BUILTIN_CFUNCTION(kBuiltinNameOrd);
  DECLARE_BUILTIN_CFUNCTION(kBuiltinNameCallable);
  DECLARE_BUILTIN_CFUNCTION(kBuiltinNameGetattr);
  DECLARE_BUILTIN_CFUNCTION(kBuiltinNameHasattr);

  // math.log
  py::object math_builtin = Utils::GetModuleAttr("math", kBuiltinNameLog, false, false);
  c_function_obj = PyCFunction_GET_FUNCTION(math_builtin.ptr());
  kBuiltinFuncOrMethodWhileList.emplace(c_function_obj);

  // python object cfunction without sideeffect
  std::map<PyObject *, std::vector<std::string>> obj_cfunc_name = {
    {py::dict().inc_ref().ptr(),
     {"__contains__", "__getitem__", "__sizeof__", "get", "keys", "items", "values", "fromkeys", "copy"}},
    {py::list().inc_ref().ptr(), {"__getitem__", "__sizeof__", "copy", "index", "count"}},
    {py::tuple().inc_ref().ptr(), {"index", "count"}},
    {py::set().inc_ref().ptr(), {"__contains__", "copy", "issubset", "__sizeof__"}},
    {py::str().inc_ref().ptr(),
     {"find",    "count",        "index",       "rfind",   "rindex",     "startswith", "endswith",  "isascii",
      "islower", "isupper",      "istitle",     "isspace", "isdecimal",  "isdigit",    "isnumeric", "isalpha",
      "isalnum", "isidentifier", "isprintable", "format",  "format_map", "__format__", "__sizeof__"}},
  };
  for (auto item : obj_cfunc_name) {
    for (auto meth : item.second) {
      py::object builtin = py::cast<py::object>(item.first).attr(meth.c_str());
      c_function_obj = PyCFunction_GET_FUNCTION(builtin.ptr());
      kBuiltinFuncOrMethodWhileList.emplace(c_function_obj);
    }
  }
  for (auto item : obj_cfunc_name) {
    Py_XDECREF(item.first);
  }
  return kBuiltinFuncOrMethodWhileList;
}

#undef DECLARE_BUILTIN_CFUNCTION

bool CheckBuiltinFuncOrMethod(const py::object &f) {
  PyObject *func = f.ptr();
  if (PyMethod_Check(func)) {
    func = PyMethod_GET_FUNCTION(func);
  }
  if (!PyCFunction_Check(func)) {
    return false;
  }
  auto c_function_obj = PyCFunction_GET_FUNCTION(func);
  if (GenCFunctionMap().find(c_function_obj) == GenCFunctionMap().end()) {
    return false;
  }
  return true;
}

static bool InferBuiltinFuncOrMethod(CallNode *call_node) {
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
  if (name == kBuiltinNameIsinstance) {
    guard_success = GuardIsInstance(call_node);
  } else {
    guard_success = GuardBuiltinFunc(call_node);
  }
  if (guard_success) {
    return CallNodeReturnConst(call_node, sub_graph, call_node->GetVobj());
  }
  return false;
}

static bool CheckTensorAsType(const py::object &func) {
  PyObject *op = func.ptr();
  if (op == nullptr) {
    return false;
  }
  if (PyMethod_Check(op)) {
    op = PyMethod_GET_FUNCTION(op);
  }
  if (!PyFunction_Check(op)) {
    return false;
  }
  auto func_ptr = reinterpret_cast<PyFunctionObject *>(op);
  std::string name = PyUnicode_AsUTF8(func_ptr->func_module);
  bool is_func = name == "mindspore.common.tensor";
  return is_func;
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

bool CheckListAppend(const py::object &func) {
  static PyCFunction append = nullptr;
  if (append == nullptr) {
    append = PyCFunction_GET_FUNCTION(py::list().attr(kBuiltinNameAppend).ptr());
  }
  PyObject *op = func.ptr();
  if (PyMethod_Check(op)) {
    op = PyMethod_GET_FUNCTION(op);
  }
  /**
   * this expression "list.append" will get type "method_descriptor"
   * this expression "[].append" will get type "built-in function"
   */
  if (!PyCFunction_Check(op)) {
    return false;
  }
  return PyCFunction_GET_FUNCTION(op) == append;
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

// special function list
// special function that mindspore support and not inline,
// the return values or type can be infer
static const std::unordered_map<std::string, SpecialAction> kFuncWhiteListMap = {
  // fuzzy match
  {kMindsporeNamePrimitive, {CheckPrimitive, InferPrimitive}},
  {kMindsporeNameMetaFuncGraph, {CheckMetaFunc_, InferMetaFunc_}},
  {kMindsporeNameGradFunc, {CheckGradFunc, InferGradFunc}},
  {kMindsporeNameMsCell, {CheckCell, InferCell}},
  // name match
  {kMindsporeNameJitFunc, {CheckJitFunc, SetCallResType<AObject::kTypeTensor>}},
  {kMindsporeNameGetCachePrim, {CheckGetCachePrim_, InferGetCachePrim_}},
  {kMindsporeNameRegistryGet, {CheckRegistryGet, InferRegistryGet}},
  {kMindsporeNameTensorInitCheck, {CheckTensorBypass, InferTensorBypass}},
  {kMindsporeNameTensorContiguous, {CheckTensorBypass, InferTensorBypass}},
  // builtin_function_or_method
  {kBuiltinNameFunctionOrMethod, {CheckBuiltinFuncOrMethod, InferBuiltinFuncOrMethod}},
  // object convert map
  {kMindsporeNameConvertMap, {CheckConvertMap, InferConvertMap}},
  {kJitForbidden, {CheckJitForbidden, SetCallResType<AObject::kTypeAnyValue>}},
  {kJitConstexpr, {CheckJitConstexpr, JustCallAndSetRes}},
  {kMindsporeNameConstexpr, {CheckMSConstexpr, InferMSConstexpr}},
  {kMindsporeNamePrimexpr, {CheckMSPrimexpr, InferMSConstexpr}},
  {kMindsporeNameTensorAsType, {CheckTensorAsType, InferTensorAsType}},
  {kBuiltinNameAppend, {CheckListAppend, InferListAppend}},
};

static const std::vector<std::pair<CheckFunc, std::string>> kFuncWhiteListFuzzyMatcher = {
  {CheckJitConstexpr, kJitConstexpr},
  {CheckMetaFunc_, kMindsporeNameMetaFuncGraph},
  {CheckGradFunc, kMindsporeNameGradFunc},
  // guard these call by short traces
  {CheckCell, kMindsporeNameMsCell},
  {CheckConvertMap, kMindsporeNameConvertMap},
  // builtin_function_or_method
  {CheckBuiltinFuncOrMethod, kBuiltinNameFunctionOrMethod},
  {CheckJitForbidden, kJitForbidden},
};

static const std::unordered_map<std::string, SpecialAction> kMindFuncWhiteListMap = {
  {kMindsporeNameJitFunc, {CheckJitFunc, SetCallResType<AObject::kTypeTensor>}},
  {kMindsporeNameGetCachePrim, {CheckGetCachePrim_, InferGetCachePrim_}},
  {kMindsporeNameRegistryGet, {CheckRegistryGet, InferRegistryGet}},
  {kMindsporeNameTensorInitCheck, {CheckTensorBypass, InferTensorBypass}},
  {kMindsporeNameTensorContiguous, {CheckTensorBypass, InferTensorBypass}},
  {kBuiltinNameFunctionOrMethod, {CheckBuiltinFuncOrMethod, InferBuiltinFuncOrMethod}},
  {kJitForbidden, {CheckJitForbidden, SetCallResType<AObject::kTypeAnyValue>}},
  {kJitConstexpr, {CheckJitConstexpr, JustCallAndSetRes}},
};

static const std::vector<std::pair<CheckFunc, std::string>> kMindFuncWhiteListFuzzyMatcher = {
  {CheckJitConstexpr, kJitConstexpr},
  {CheckBuiltinFuncOrMethod, kBuiltinNameFunctionOrMethod},
  {CheckJitForbidden, kJitForbidden},
};

const std::string GetMindsporeNamePrimitive() { return kMindsporeNamePrimitive; }

const std::unordered_map<std::string, SpecialAction> &GetFuncWhiteListMap(bool trace_flag) {
  if (trace_flag) {
    return kMindFuncWhiteListMap;
  } else {
    return kFuncWhiteListMap;
  }
}
const std::vector<std::pair<CheckFunc, std::string>> &GetFuncWhiteListFuzzyMatcher(bool trace_flag) {
  if (trace_flag) {
    return kMindFuncWhiteListFuzzyMatcher;
  } else {
    return kFuncWhiteListFuzzyMatcher;
  }
}
}  // namespace pijit
}  // namespace mindspore
