
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
#include <string>
#include <memory>
#include <algorithm>
#include <utility>
#include <unordered_map>
#include <vector>
#include <set>
#include "pipeline/jit/pi/common.h"
#include "pipeline/jit/pi/external.h"
#include "pipeline/jit/pi/graph_capture/graph_build.h"
#include "pipeline/jit/pi/graph_guard/infer.h"

namespace mindspore {
namespace jit {
namespace graph {
using CheckFunc = bool (*)(const py::object &);
using InferFunc = bool (*)(CallNode *);
struct SpecialAction {
  CheckFunc check;
  InferFunc infer;
};

extern AObject *InferFuncResult(const py::object &func, const std::vector<AObject *> &stack_args, int opcode,
                                const GraphJitConfig &conf, bool clear_guard);

extern AObject *InferFuncResult(const py::object &func, const py::object &args, const py::object &kwargs,
                                const GraphJitConfig &conf, bool clear_guard);

// ------------------------------builtins functions--------------------------------
static constexpr const char *kBuiltinNameIsinstance = "isinstance";  // call __instancecheck__
static constexpr const char *kBuiltinNameIssubclass = "issubclass";  // call __subclasscheck__
static constexpr const char *kBuiltinNameLen = "len";                // call __len__
static constexpr const char *kBuiltinNameAbs = "abs";                // call __abs__
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
// static constexpr const char *kBuiltinNameAppend = "append";  // list update
// ------------------------------builtins method--------------------------------

// ------------------------------mindspore functions-------------------------------
static constexpr const char *kMindsporeNameGetCachePrim = "_get_cache_prim";
static constexpr const char *kMindsporeNameRegistryGet = "get";
static constexpr const char *kMindsporeNameConstexpr = "CompileOp";
/**
 * NOTE: mindspore/ops/composite/base.py, after_grad decorated by '_warp_func'
 * code name is 'wrapper', not 'after_grad', it only called by pynative
 */
static constexpr const char *kMindsporeNameGradFunc = "after_grad";
static constexpr const char *kMindsporeNameJitFunc = "staging_specialize";  // mindspore.jit

static constexpr const char *kMindsporeNamePrimitive = "Primitive_";
static constexpr const char *kMindsporeNameMetaFuncGraph = "MetaFuncGraph_";
static constexpr const char *kMindsporeNameMsCell = "mindspore.nn.Cell";
/**
 * convert function map
 * refer to convert_object_map in mindspore._extends.parse.resources.py
 */
static constexpr const char *kMindsporeNameConvertMap = "mindspore._extends.parse.resources.convert_object_map";
// ------------------------------mindspore functions-------------------------------

static constexpr const char *kJitForbidden = ".pijit_forbidden";
static constexpr const char *kJitConstexpr = ".pijit_constexpr";

static py::object GetGradClass() { return Utils::GetModuleAttr("mindspore._c_expression", "GradOperation_"); }

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
    MS_LOG(ERROR) << "got an error while call the <pijit.constexpr> " << std::string(py::str(func.ptr())) << " "
                  << py::error_already_set().what();
    PyErr_Clear();
  }
  call_node->SetVobj(AObject::Convert(value));
  call_node->SetSubGraph(nullptr);
  return false;
}

bool CallNodeReturnConst(CallNode *call_node, Graph *sub_graph, AObject *value, const std::string &global_key = "") {
  static const std::set<AObject::Type> cnst_types = {
    AObject::kTypeBool,  AObject::kTypeInt,  AObject::kTypeString,   AObject::kTypeTuple,
    AObject::kTypeFloat, AObject::kTypeNone, AObject::kTypeEllipsis,
  };
  auto &alloc = sub_graph->allocator();
  ValueNode *ret_node;
  if (global_key.empty()) {
    if (cnst_types.find(value->GetType()) == cnst_types.end()) {
      // NOTE: python gc can't check code object reference, const object shouldn't reference other none const object
      MS_LOG(DEBUG) << value->ToString() + " as const is unsupported";
      return false;
    }
    ret_node = alloc.NewValueNode(value, LOAD_CONST, -1, {});
  } else {
    ret_node = alloc.NewValueNode(value, LOAD_GLOBAL, -1, {});
    ret_node->SetName(global_key);
    call_node->GetGraph()->InstallToGlobal(global_key, value->GetPyObject());
  }
  call_node->SetSubGraph(sub_graph);
  ret_node->SetGraph(call_node->GetGraph());

  sub_graph->AddInstr(ret_node);
  sub_graph->AddInstr(alloc.NewInstrNode(RETURN_VALUE, 0));
  sub_graph->SetRetVal(ret_node);
  call_node->SetInlineReason(InlineReason::kInline);

  AbstractNodeList b;
  for (auto i = call_node->getInputs().size(); i > 0; --i) {
    b.push_back(call_node->GetGraph()->allocator().NewInstrNode(POP_TOP, 0));
  }
  call_node->SetExtraOper(reinterpret_cast<InstrNode *>(b.head()));
  return true;
}

bool GuardConstCallNodeParam(CallNode *call_node, Graph *sub_graph, int max_guard_depth) {
  std::vector<std::pair<TracePtr, GuardLevel>> traces;
  for (auto i : call_node->getInputs()) {
    if (i->GetOpcode() == LOAD_CONST) {
      continue;
    }
    AObject::Type type = i->GetVobj() ? i->GetVobj()->GetType() : AObject::kTypeAnyValue;
    if (type == AObject::kTypeAnyValue) {
      return false;
    }
    TracePtr tr = sub_graph->TraceValueNode(i, max_guard_depth);
    if (tr == nullptr) {
      return false;
    }
    GuardLevel level = GuardLevel::GEqual;
    if (type == AObject::kTypeTensor) {
      if (i->GetOpcode() == LOAD_GLOBAL) {
        level = GuardLevel::GId;  // only guard global tensor
      } else {
        return false;
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
  return true;
}

static bool check_ConvertMap(const py::object &func) {
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

static bool infer_ConvertMap(CallNode *call_node) {
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
    auto inst = mindspore::jit::graph::InferEngine::GetInstance();
    bool is_abstract = false;
    PyObject *ret = inst->InferPrimitive(infer_obj.ptr(), list, &is_abstract);
    if (ret == nullptr) {
      return false;
    }
    AObject::Type type = AObject::GetPyType(ret);
    res = is_abstract ? AObject::MakeAObject(type) : AObject::Convert(ret);
    Py_DECREF(ret);
  } else {
    return false;
  }
  if (res) {
    call_node->SetVobj(res);
  }
  return false;
}

bool check__get_cache_prim(const py::object &f) {
  if (!PyFunction_Check(f.ptr())) {
    return false;
  }
  auto func_ptr = reinterpret_cast<PyFunctionObject *>(f.ptr());
  std::string name = PyUnicode_AsUTF8(func_ptr->func_module);
  bool is_func = name == "mindspore.ops._primitive_cache";
  return is_func;
}

bool infer__get_cache_prim(CallNode *n) {
  // just return the first parameter of _get_cache_prim
  Graph *g = n->GetSubGraph();
  n->SetVobj(n->input(1)->GetVobj());
  g->SetRetVal(n->input(1));

  // extra operation
  auto &alloc = g->allocator();
  AbstractNodeList b = {nullptr, nullptr};
  b.push_back(alloc.NewInstrNode(ROT_TWO, 0));
  b.push_back(alloc.NewInstrNode(POP_TOP, 0));
  n->SetExtraOper(reinterpret_cast<InstrNode *>(b.head()));
  return true;
}

static bool check_RegistryGet(const py::object &func) {
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

static bool infer_RegistryGet(CallNode *call_node) {
  Graph *g = call_node->GetSubGraph();
  JustCallAndSetRes(call_node);

  py::object func = call_node->GetVobj()->GetPyObject();
  if (call_node->getInputs().back()->GetOpcode() == LOAD_CONST && func.ptr() != nullptr) {
    // constant function call
    std::string key = py::str(func.ptr());
    return CallNodeReturnConst(call_node, g, call_node->GetVobj(), key);
  }
  return false;
}

static bool builtins_module_check(PyObject *m) {
  return m && PyModule_Check(m) && !strcmp(PyModule_GetName(m), "builtins");
}

bool check_builtin_cfunc(const py::object &f) {
  PyObject *func = f.ptr();
  if (PyMethod_Check(func)) {
    func = PyMethod_GET_FUNCTION(func);
  }
  if (!PyCFunction_Check(func)) {
    return false;
  }
  return builtins_module_check(reinterpret_cast<PyCFunctionObject *>(func)->m_self);
}

bool infer_builtin_len(CallNode *n) {
  auto g = n->GetSubGraph();
  n->SetSubGraph(nullptr);
  n->SetVobj(AObject::MakeAObject(AObject::kTypeInt));
  AObject *arg = n->input(1)->GetVobj();
  if (!arg) {
    return false;
  }
  Py_ssize_t siz = 0;
  switch (arg->GetType()) {
    case AObject::kTypeTuple:
    case AObject::kTypeList:
      siz = static_cast<AbstractTuple *>(arg)->size();
      break;
    case AObject::kTypeDict:
      siz = static_cast<AbstractDict *>(arg)->size();
      break;
    default:
      if (!arg->GetPyObject().ptr()) {
        return false;
      }
      siz = PyObject_Size(arg->GetPyObject().ptr());
      break;
  }
  if (siz < 0) {
    PyErr_Clear();
    return false;
  }
  AObject *res = AObject::Convert(py::int_(siz));
  n->SetVobj(res);
  if (g->GuardValueNode(n)) {
    return CallNodeReturnConst(n, g, res);
  }
  return false;
}

bool infer_builtin_getattr(CallNode *call_node) {
  call_node->SetSubGraph(nullptr);
  ValueNode *load_name = call_node->input(2);
  PyObject *pyname = load_name->GetVobj()->GetPyObject().ptr();
  if (!pyname || !PyUnicode_Check(pyname)) {
    // has a python exceptions, do nothing
    return false;
  }
  const char *name = PyUnicode_AsUTF8(pyname);
  AObject *attr = call_node->input(1)->get_attr(name);
  call_node->SetVobj(attr);
  return false;
}

static inline bool InferBuiltinOneArg(CallNode *call_node, PyCFunction cpython_func) {
  auto &arg = call_node->input(1)->GetVobj();
  if (arg && arg->GetPyObject().ptr() && arg->GetType() != AObject::kTypeAnyValue) {
    py::object res = py::reinterpret_steal<py::object>(cpython_func(nullptr, arg->GetPyObject().ptr()));
    call_node->SetVobj(AObject::Convert(res));
    PyErr_Clear();
  }
  call_node->SetSubGraph(nullptr);
  return false;
}

#define DECLARE_BUILTIN_CFUNCTION(func_name)                             \
  static PyCFunction cpython_func = nullptr;                             \
  if (!cpython_func) {                                                   \
    PyObject *p = PyDict_GetItemString(PyEval_GetBuiltins(), func_name); \
    MS_ASSERT(p &&PyCFunction_Check(p));                                 \
    cpython_func = PyCFunction_GET_FUNCTION(p);                          \
  }

#define DECLARE_INFER_BUILTIN_ONE_ARG(func_name) \
  [](CallNode *n) {                              \
    DECLARE_BUILTIN_CFUNCTION(func_name);        \
    return InferBuiltinOneArg(n, cpython_func);  \
  }

using InstanceSubclassCheckFunc = int (*)(PyObject *, PyObject *);
template <InstanceSubclassCheckFunc pyfunc>
bool InferBuiltinInstanceSubclassCheck(CallNode *call_node) {
  Graph *g = call_node->GetSubGraph();
  call_node->SetVobj(AObject::MakeAObject(AObject::kTypeBool));
  call_node->SetSubGraph(nullptr);
  auto &arg1 = call_node->input(1)->GetVobj();
  auto &arg2 = call_node->input(2)->GetVobj();
  if (arg1 == nullptr || arg2 == nullptr || arg1->GetPyObject().ptr() == nullptr ||
      arg2->GetPyObject().ptr() == nullptr) {
    return false;
  }
  int stat = pyfunc(arg1->GetPyObject().ptr(), arg2->GetPyObject().ptr());
  if (stat < 0) {
    PyErr_Clear();
    return false;
  }
  AObject *res = AObject::Convert(py::bool_(stat));
  call_node->SetVobj(res);
  if (g->GuardValueNode(call_node)) {
    return CallNodeReturnConst(call_node, g, res);
  }
  return false;
}

static bool support_infer_primitive(PyObject *obj) {
  if (obj == nullptr) {
    return false;
  }
  if (IsPrimitiveType<true>(Py_TYPE(obj))) {
    auto inst = mindspore::jit::graph::InferEngine::GetInstance();
    return inst->SupportInfer(obj);
  }
  return false;
}

static bool check_primitive(const py::object &func) {
  return AObject::GetPyType(func.ptr()) == AObject::kTypePrimitive;
}

bool infer_primitive(CallNode *call_node) {
  static const std::unordered_map<std::string, AObject::Type> not_ret_tensor_prim = {
    {"Prim[_get_grad_op]<constexpr_prim=True>", AObject::kTypeMetaFuncGraph},
    {"Prim[DType]", AObject::kTypeAnyValue},
    {"Prim[Partial]<side_effect_propagate=1>", AObject::kTypeAnyValue},
  };
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
  if (!support_infer_primitive(prim)) {
    return false;
  }

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

  auto inst = mindspore::jit::graph::InferEngine::GetInstance();
  bool is_abstract = false;
  PyObject *ret;
  try {
    ret = inst->InferPrimitive(prim, list, &is_abstract);
  } catch (std::exception &e) {
    MS_LOG(INFO) << "infer primitive failed. reason:";
    MS_LOG(INFO) << e.what();
    ret = nullptr;
  }
  if (ret == nullptr) {
    return false;
  }
  AObject::Type type = AObject::GetPyType(ret);
  AObject *type_info = is_abstract ? AObject::MakeAObject(type) : AObject::Convert(ret);
  call_node->SetVobj(type_info);
  Py_DECREF(ret);
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

static bool check_MetaFunc_(const py::object &o) {
  PyTypeObject *tp = PyType_Check(o.ptr()) ? reinterpret_cast<PyTypeObject *>(o.ptr()) : Py_TYPE(o.ptr());
  return IsMetaFuncGraphType<true>(tp);
}

static bool infer_MetaFunc_(CallNode *call_node) {
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
  guard->GuardOn(*trace, mindspore::jit::graph::GuardLevel::GEqual);
  if (config.GetBoolConfig(GraphJitConfig::kGuardDetachObject)) {
    (*trace)->Detach();
  }
  call_node->SetSubGraph(nullptr);
  HandleGradFuncCall(call_node, AObject::Convert(decorated_func), sens_param);
}

static bool check_GradFunc(const py::object &f) {
  if (!PyFunction_Check(f.ptr())) {
    return false;
  }
  std::string decorated_name = PyUnicode_AsUTF8(reinterpret_cast<PyFunctionObject *>(f.ptr())->func_qualname);
  return decorated_name == "_Grad.__call__.<locals>.after_grad" ||
         decorated_name == "GradOperation.__call__.<locals>.after_grad";
}

static bool infer_GradFunc(CallNode *call_node) {
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

static bool check_JitFunc(const py::object &o) {
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

static bool check_Cell(const py::object &callable_info) {
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

static bool infer_Cell(CallNode *call_node) {
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

static bool check_JitForbidden(const py::object &func) {
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

static bool check_JitConstexpr(const py::object &func) { return kPIJitConfigDefault.CheckJitConstexpr(func); }
static bool check_MSConstexpr(const py::object &func) {
  std::string tp_name = py::str(reinterpret_cast<PyObject *>(Py_TYPE(func.ptr())));
  constexpr const char name[] = ".<locals>.deco.<locals>.CompileOp'>";
  constexpr const int size = sizeof(name) - 1;
  return tp_name.size() > size ? !tp_name.compare(tp_name.size() - size, size, name) : false;
}

static bool infer_MSConstexpr(CallNode *call_node) {
  Graph *g = call_node->GetSubGraph();
  JustCallAndSetRes(call_node);

  py::object cnst = call_node->GetVobj()->GetPyObject();
  if (cnst.ptr() == nullptr) {
    return false;
  }
  if (!GuardConstCallNodeParam(call_node, g, 2)) {
    return false;
  }

  return CallNodeReturnConst(call_node, g, call_node->GetVobj());
}

// special function list
// special function that mindspore support and not inline,
// the return values or type can be infer
static const std::unordered_map<std::string, SpecialAction> kFuncWhiteListMap = {
  // fuzzy match
  {kMindsporeNamePrimitive, {check_primitive, infer_primitive}},
  {kMindsporeNameMetaFuncGraph, {check_MetaFunc_, infer_MetaFunc_}},
  {kMindsporeNameGradFunc, {check_GradFunc, infer_GradFunc}},
  {kMindsporeNameMsCell, {check_Cell, infer_Cell}},
  // name match
  {kMindsporeNameJitFunc, {check_JitFunc, SetCallResType<AObject::kTypeTensor>}},
  {kMindsporeNameGetCachePrim, {check__get_cache_prim, infer__get_cache_prim}},
  {kMindsporeNameRegistryGet, {check_RegistryGet, infer_RegistryGet}},
  {kBuiltinNameLen, {check_builtin_cfunc, infer_builtin_len}},
  {kBuiltinNameAbs, {check_builtin_cfunc, DECLARE_INFER_BUILTIN_ONE_ARG(kBuiltinNameAbs)}},
  // NOTE: call __bool__ hook for each item
  {kBuiltinNameAll, {check_builtin_cfunc, DECLARE_INFER_BUILTIN_ONE_ARG(kBuiltinNameAll)}},
  // NOTE: call __bool__ hook for each item
  {kBuiltinNameAny, {check_builtin_cfunc, DECLARE_INFER_BUILTIN_ONE_ARG(kBuiltinNameAny)}},
  {kBuiltinNameHash, {check_builtin_cfunc, DECLARE_INFER_BUILTIN_ONE_ARG(kBuiltinNameHash)}},
  {kBuiltinNameIsinstance, {check_builtin_cfunc, InferBuiltinInstanceSubclassCheck<PyObject_IsInstance>}},
  {kBuiltinNameIssubclass, {check_builtin_cfunc, InferBuiltinInstanceSubclassCheck<PyObject_IsSubclass>}},
  {kBuiltinNameId, {check_builtin_cfunc, SetCallResType<AObject::kTypeInt>}},
  {kBuiltinNameOrd, {check_builtin_cfunc, DECLARE_INFER_BUILTIN_ONE_ARG(kBuiltinNameOrd)}},
  {kBuiltinNameCallable, {check_builtin_cfunc, DECLARE_INFER_BUILTIN_ONE_ARG(kBuiltinNameCallable)}},
  {kBuiltinNameGetattr, {check_builtin_cfunc, infer_builtin_getattr}},
  {kBuiltinNameHasattr, {check_builtin_cfunc, SetCallResType<AObject::kTypeBool>}},
  // object convert map
  {kMindsporeNameConvertMap, {check_ConvertMap, infer_ConvertMap}},
  {kJitForbidden, {check_JitForbidden, SetCallResType<AObject::kTypeAnyValue>}},
  {kJitConstexpr, {check_JitConstexpr, JustCallAndSetRes}},
  {kMindsporeNameConstexpr, {check_MSConstexpr, infer_MSConstexpr}},
};

static const std::vector<std::pair<CheckFunc, std::string>> kFuncWhiteListFuzzyMatcher = {
  {check_JitConstexpr, kJitConstexpr},
  {check_MetaFunc_, kMindsporeNameMetaFuncGraph},
  {check_GradFunc, kMindsporeNameGradFunc},
  // guard these call by short traces
  {check_Cell, kMindsporeNameMsCell},
  {check_ConvertMap, kMindsporeNameConvertMap},
  {check_JitForbidden, kJitForbidden},
};

static const char *GetFuncName(const py::object &f) {
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

bool IsFuncInWhiteList(const py::object &f, std::string *special_func_key, bool bInferPrimitive) {
  if (f.ptr() == nullptr) {
    return false;
  }
  *special_func_key = GetFuncName(f);
  auto iter = kFuncWhiteListMap.find(*special_func_key);
  if (iter != kFuncWhiteListMap.end()) {
    return iter->second.check(f);
  }
  auto tar = std::find_if(kFuncWhiteListFuzzyMatcher.begin(), kFuncWhiteListFuzzyMatcher.end(),
                          [&f](const std::pair<CheckFunc, std::string> &i) { return i.first(f); });
  if (tar != kFuncWhiteListFuzzyMatcher.end()) {
    *special_func_key = tar->second;
    return true;
  }
  if (bInferPrimitive && check_primitive(f)) {
    *special_func_key = kMindsporeNamePrimitive;
    return true;
  }
  return false;
}

bool HandleFuncInWhiteList(const std::string &key, CallNode *n) {
  MS_LOG(DEBUG) << "specialize for " << key;
  return kFuncWhiteListMap.find(key)->second.infer(n);
}
}  // namespace graph
}  // namespace jit
}  // namespace mindspore
