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
#include "pipeline/jit/pi/graph_guard/trace.h"
#include <map>
#include <vector>
#include <unordered_map>
#include <functional>
#include <utility>
#include "pipeline/jit/pi/graph_guard/guard_utils.h"
#include "pybind11/pybind11.h"
#include "pybind_api/ir/primitive_py.h"
#include "include/common/utils/convert_utils_py.h"
#include "pipeline/jit/pi/graph_guard/infer.h"
#include "pipeline/jit/pi/utils/utils.h"
#include "include/common/utils/python_adapter.h"

namespace mindspore {
namespace jit {
namespace graph {

extern bool check_builtin_cfunc(const py::object &func);

Trace::Trace(PyObject *pObj, std::shared_ptr<Trace> pOrigin) : obj_(pObj), origin_(pOrigin) {
  if (pOrigin != nullptr) {
    originType_ = pOrigin->GetOriginType();
    curType_ = pOrigin->GetTraceType();
  } else {
    originType_ = Unknown;
    curType_ = Unknown;
  }
  if (obj_ != Py_None && obj_ != NULL) {
    Py_INCREF(obj_);
  }
}

Trace::~Trace() {
  if (obj_ != Py_None && obj_ != NULL) {
    Py_DECREF(obj_);
  }
}

TracePtr Trace::GetOrigin() {
  if (origin_ != nullptr) {
    return origin_;
  } else {
    return nullptr;
  }
}

PyObject *Trace::GetObject() { return obj_; }

TraceType Trace::GetTraceType() { return curType_; }

TraceType Trace::GetOriginType() { return originType_; }

void Trace::Replace(std::shared_ptr<Trace> dst, std::shared_ptr<Trace> src) {
  if (origin_ != nullptr) {
    if (*origin_ == *src) {
      origin_ = dst;
    } else {
      origin_->Replace(dst, src);
    }
  }
}

bool Trace::operator==(const Trace &trace) {
  if (curType_ == trace.curType_ && obj_ == trace.obj_) {
    return true;
  } else {
    return false;
  }
}

void Trace::Detach() {
  if (obj_ != Py_None && obj_ != nullptr) {
    Py_DECREF(obj_);
    obj_ = nullptr;
  }
  if (origin_ != nullptr) {
    origin_->Detach();
  }
}

RootTrace::RootTrace(PyObject *pObj, TraceType tt, int index, std::string name, std::string module_name)
    : Trace(pObj, nullptr), idx_(index), name_(name), module_name_(module_name) {
  originType_ = tt;
  curType_ = tt;
}

void RootTrace::GetParam(int *index, std::string *name, std::string *module_name) {
  *index = idx_;
  *name = name_;
  *module_name = module_name_;
}

PyObject *RootTrace::Retrieve(PTraceContext context) {
  PyObject *ret = nullptr;
  switch (curType_) {
    case TraceType::Global: {
      MS_EXCEPTION_IF_CHECK_FAIL(name_.size() > 0, "check trace");
      PyObject *globals = context->f_globals;
      if (!module_name_.empty()) {
        PyObject *mn = PyUnicode_FromString(module_name_.c_str());
        PyObject *mm = PyImport_GetModule(mn);  // ensure module is initialized
        if (mn != nullptr && mm != nullptr) {
          globals = PyModule_GetDict(mm);
        }
        PyErr_Clear();
        Py_XDECREF(mn);
        Py_XDECREF(mm);
      }
      PyObject *key = PyUnicode_FromString(name_.c_str());
      ret = PyObject_GetItem(globals, key);
      if (ret == nullptr) {
        PyErr_Clear();
        ret = PyObject_GetItem(context->f_builtins, key);
        if (ret == nullptr) {
          PyErr_Clear();
        }
      }
      Py_DECREF(key);
      return ret;
    }
    case TraceType::Deref: {
      PyObject *cell = context->f_localsplus[context->f_code->co_nlocals + idx_];
      if (cell != nullptr && cell != Py_None) {
        ret = PyCell_GET(cell);
        Py_XINCREF(ret);
      }
      return ret;
    }
    case TraceType::Closure: {
      ret = context->f_localsplus[context->f_code->co_nlocals + idx_];
      Py_XINCREF(ret);
      return ret;
    }
    case TraceType::BuiltIn: {
      MS_EXCEPTION_IF_CHECK_FAIL(name_.size() > 0, "check trace");
      PyObject *key = PyUnicode_FromString(name_.c_str());
      ret = PyObject_GetItem(context->f_builtins, key);
      if (ret == nullptr) {
        PyErr_Clear();
        ret = PyObject_GetItem(context->f_globals, key);
        if (ret == nullptr) {
          PyErr_Clear();
        }
      }
      Py_DECREF(key);
      return ret;
    }
    case TraceType::Local:
      ret = context->f_locals;
      break;
    case TraceType::Param:
      ret = context->f_localsplus[idx_];
      break;
    default:
      break;
  }
  if (ret != Py_None && ret != NULL) {
    Py_INCREF(ret);
  }
  return ret;
}

std::string RootTrace::ToString() {
  std::string ret;
  switch (curType_) {
    case TraceType::Global:
      if (!module_name_.empty()) {
        ret = "(global " + module_name_ + ".__dict__[" + name_ + "])";
      } else {
        ret = "f_globals[" + name_ + "]";
      }
      break;
    case TraceType::Deref:
      ret = "f_freevars[" + std::to_string(idx_) + "]";
      break;
    case TraceType::Closure:
      ret = "f_closure[" + std::to_string(idx_) + "]";
      break;
    case TraceType::BuiltIn:
      ret = "f_builtins";
      break;
    case TraceType::Local:
      ret = "f_locals";
      break;
    case TraceType::Param:
      ret = "f_localsplus[";
      ret += std::to_string(idx_);
      ret += "]";
      break;
    default:
      ret = "unknown_root";
      break;
  }
  return ret;
}

bool RootTrace::operator==(const Trace &trace) {
  bool ret = false;
  if (Trace::operator==(trace)) {
    const RootTrace &t = (const RootTrace &)trace;
    ret = idx_ == t.idx_;
    if (ret && idx_ == -1) {
      ret = name_ == t.name_ && module_name_ == t.module_name_;
    }
  }
  return ret;
}

ItemTrace::ItemTrace(PyObject *pObj, TracePtr pOrigin, TracePtr pItem) : Trace(pObj, pOrigin), item_(pItem) {
  curType_ = TraceType::Item;
}

TracePtr ItemTrace::GetItem() { return item_; }

void ItemTrace::Replace(std::shared_ptr<Trace> dst, std::shared_ptr<Trace> src) {
  Trace::Replace(dst, src);
  if (item_ != nullptr) {
    if (*item_ == *src) {
      item_ = dst;
    } else {
      item_->Replace(dst, src);
    }
  }
}

PyObject *ItemTrace::Retrieve(PTraceContext context) {
  PyObject *ret = NULL;
  if (origin_ != nullptr && item_ != nullptr) {
    PyObject *pSet = origin_->Retrieve(context);
    PyObject *pItem = item_->Retrieve(context);
    if (pSet != NULL && pItem != NULL) {
      if (PyDict_CheckExact(pSet)) {
        ret = PyDict_GetItem(pSet, pItem);
        Py_INCREF(ret);
      } else {
        ret = PyObject_GetItem(pSet, pItem);
      }
      Py_DECREF(pSet);
      Py_DECREF(pItem);
    }
  }
  return ret;
}

std::string ItemTrace::ToString() {
  std::string ret;
  if (origin_ != nullptr && item_ != nullptr) {
    std::string ori = origin_->ToString();
    std::string itm = item_->ToString();
    ret = ori + "[" + itm + "]";
  }
  return ret;
}

bool ItemTrace::operator==(const Trace &trace) {
  if (Trace::operator==(trace)) {
    const ItemTrace &t = (const ItemTrace &)trace;
    if (!item_ && !(t.item_)) {
      return true;
    } else if (item_ != nullptr && t.item_ != nullptr) {
      return *item_ == *(t.item_);
    }
  }
  return false;
}

void ItemTrace::Detach() {
  Trace::Detach();
  if (item_ != nullptr) {
    item_->Detach();
  }
}

AttrTrace::AttrTrace(PyObject *pObj, TracePtr pOrigin, std::string strAttr) : Trace(pObj, pOrigin), attr_(strAttr) {
  curType_ = TraceType::Attr;
}

std::string AttrTrace::GetAttribute() { return attr_; }

PyObject *AttrTrace::Retrieve(PTraceContext context) {
  PyObject *ret = NULL;
  if (origin_ != nullptr) {
    PyObject *pOrigin = origin_->Retrieve(context);
    if (pOrigin != NULL) {
      PyObject *itemName = PyUnicode_FromString(attr_.c_str());
      if (PyDict_CheckExact(pOrigin)) {
        ret = PyDict_GetItem(pOrigin, itemName);
        Py_INCREF(ret);
      } else {
        ret = PyObject_GetItem(pOrigin, itemName);
      }
      Py_DECREF(itemName);
      Py_DECREF(pOrigin);
    }
  }
  return ret;
}

std::string AttrTrace::ToString() {
  std::string ret;
  if (origin_ != nullptr) {
    std::string ori = origin_->ToString();
    ret = ori + "." + attr_;
  }
  return ret;
}

bool AttrTrace::operator==(const Trace &trace) {
  if (Trace::operator==(trace)) {
    const AttrTrace &t = (const AttrTrace &)trace;
    return attr_ == t.attr_;
  }
  return false;
}

ConstTrace::ConstTrace(PyObject *pObj, int iIndex) : Trace(pObj, nullptr), index_(iIndex) {
  curType_ = TraceType::Const;
  originType_ = TraceType::Const;
}

int ConstTrace::GetIndex() { return index_; }

PyObject *ConstTrace::Retrieve(PTraceContext context) {
  PyObject *ret = NULL;
  if (obj_ != NULL) {
    Py_INCREF(obj_);
    return obj_;
  }
  if (index_ >= 0 && index_ < PyTuple_GET_SIZE(context->f_code->co_consts)) {
    ret = PyTuple_GET_ITEM(context->f_code->co_consts, index_);
    Py_INCREF(ret);
  } else {
    ret = obj_;
    Py_INCREF(ret);
  }
  return ret;
}

void OpTrace::Replace(std::shared_ptr<Trace> dst, std::shared_ptr<Trace> src) {
  Trace::Replace(dst, src);
  for (size_t i = 0; i < params_.size(); ++i) {
    if (*params_[i] == *src) {
      params_[i] = dst;
    } else {
      params_[i]->Replace(dst, src);
    }
  }
}

void OpTrace::Detach() {
  Trace::Detach();
  for (auto t : params_) {
    t->Detach();
  }
}

std::string ConstTrace::ToString() {
  std::string ret = "co_consts";
  return ret + "[" + std::to_string(index_) + "]";
}

bool ConstTrace::operator==(const Trace &trace) {
  if (Trace::operator==(trace)) {
    const ConstTrace &t = (const ConstTrace &)trace;
    return index_ == t.index_;
  }
  return false;
}

void ConstTrace::Detach() {}

TypeTrace::TypeTrace(PyObject *pObj, TracePtr pOrigin) : Trace(pObj, pOrigin) {
  pType_ = Py_TYPE(pObj);
  curType_ = TraceType::Type;
}

PyTypeObject *TypeTrace::GetType() { return pType_; }

PyObject *TypeTrace::Retrieve(PTraceContext context) {
  if (origin_ != NULL) {
    PyObject *pOrigin = origin_->Retrieve(context);
    if (pOrigin != NULL) {
      return reinterpret_cast<PyObject *>(Py_TYPE(pOrigin));
    }
  }
  return NULL;
}

std::string TypeTrace::ToString() {
  std::string ret = "type";
  if (origin_ != NULL) {
    ret += "(" + origin_->ToString() + ")";
  }
  return ret;
}

bool TypeTrace::operator==(const Trace &trace) {
  if (Trace::operator==(trace)) {
    const TypeTrace &t = (const TypeTrace &)trace;
    return pType_ == t.pType_;
  }
  return false;
}

static PyObject *RichCompare(PyObject *left, PyObject *right, int oparg) {
#if (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 9)
  int stat;
  switch (oparg) {
    case PyCmp_IS:
      return left == right ? Py_True : Py_False;
    case PyCmp_IS_NOT:
      return left != right ? Py_True : Py_False;
    case PyCmp_IN:
    case PyCmp_NOT_IN:
      stat = PySequence_Contains(left, right);
      if (stat < 0) {
        return nullptr;
      }
      stat = oparg == PyCmp_IN ? stat : !stat;
      return (stat) ? Py_True : Py_False;
    case PyCmp_EXC_MATCH:
      return nullptr;
    default:
      break;
  }
#endif
  return PyObject_RichCompare(left, right, oparg);
}

static bool support_infer_primitive(PyObject *obj) {
  if (py::isinstance<mindspore::PrimitivePyAdapter>(obj)) {
    auto inst = mindspore::jit::graph::InferEngine::GetInstance();
    return inst->SupportInfer(obj);
  } else {
    return false;
  }
}

static bool support_create_primitive(PyObject *obj) {
  if (!obj || !PyType_Check(obj)) {
    return false;
  }
  py::object m = py::reinterpret_steal<py::object>(PyImport_GetModule(py::str("mindspore.ops").ptr()));
  if (!m.ptr()) {
    PyErr_Clear();
    return false;
  }
  py::object t = py::reinterpret_steal<py::object>(PyObject_GetAttrString(m.ptr(), "Primitive"));
  if (PyType_IsSubtype(reinterpret_cast<PyTypeObject *>(obj), reinterpret_cast<PyTypeObject *>((t.ptr())))) {
    return true;
  } else {
    return false;
  }
}

static bool SupportCall(PyObject *func, const std::string &name) {
  return support_infer_primitive(func) || support_create_primitive(func) || IsMsClass(func) ||
         (name.size() != 0 && PyDict_GetItemString(PyEval_GetBuiltins(), name.c_str()) == func);
}

static PyObject *DoCall(const std::vector<PyObject *> &params, int op, std::string name) {
  if (!Utils::IsCallOp(op) || params.size() < 1) {
    return nullptr;
  }
  if (support_infer_primitive(params[0])) {
    std::vector<PyObject *> list;
    auto inst = mindspore::jit::graph::InferEngine::GetInstance();
    list.insert(list.begin(), params.begin() + 1, params.end());
    bool is_abstract = false;
    return inst->InferPrimitive(params[0], list, &is_abstract);
  }
  if (!support_create_primitive(params[0]) && !IsMsClass(params[0])) {
    // current only for builtin func
    MS_EXCEPTION_IF_CHECK_FAIL(PyDict_GetItemString(PyEval_GetBuiltins(), name.c_str()) == params[0],
                               "not implement guard none builtin function");
  }

  size_t nargs = (params.size() - 1);
  size_t kw_cnt = 0;
  switch (op) {
    case CALL_FUNCTION:
      return PyObject_Vectorcall(params[0], params.data() + 1, nargs, NULL);
    case CALL_FUNCTION_KW:
      kw_cnt = PyTuple_GET_SIZE(params.back());
      return PyObject_Vectorcall(params[0], params.data() + 1, nargs - 1 - kw_cnt, params.back());
    case CALL_FUNCTION_EX:
      return PyObject_Call(params[0], params[1], params.size() > 2 ? params[2] : nullptr);
    default:
      break;
  }
  return nullptr;
}

using PyObjectArray = std::vector<PyObject *>;
using PythonBytecodeSupportCheckFunc = std::function<bool(int opargs, const PyObjectArray &objs)>;
using PythonBytecodeExecuteFunc = std::function<PyObject *(int opargs, const PyObjectArray &objs)>;
using PythonBytecodeFuncSet = std::pair<PythonBytecodeSupportCheckFunc, PythonBytecodeExecuteFunc>;
static bool ByteCodeUnsupported(int opargs, const PyObjectArray &objs) { return false; }
static bool ByteCodeSupported(int opargs, const PyObjectArray &objs) { return true; }
static std::unordered_map<int, PythonBytecodeFuncSet> kBytecodeExecuter = {
  {POP_TOP, {ByteCodeUnsupported, nullptr}},
  {ROT_TWO, {ByteCodeUnsupported, nullptr}},
  {ROT_THREE, {ByteCodeUnsupported, nullptr}},
  {DUP_TOP, {ByteCodeUnsupported, nullptr}},
  {DUP_TOP_TWO, {ByteCodeUnsupported, nullptr}},
  {NOP, {ByteCodeUnsupported, nullptr}},
  {UNARY_POSITIVE,
   {ByteCodeSupported, [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_Positive(objs[0]); }}},
  {UNARY_NEGATIVE,
   {ByteCodeSupported, [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_Negative(objs[0]); }}},
  {UNARY_NOT,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * {
      auto ret = PyObject_IsTrue(objs[0]) ? Py_True : Py_False;
      Py_INCREF(ret);
      return ret;
    }}},
  {UNARY_INVERT,
   {ByteCodeSupported, [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_Invert(objs[0]); }}},
  {BINARY_MATRIX_MULTIPLY,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_MatrixMultiply(objs[0], objs[1]); }}},
  {INPLACE_MATRIX_MULTIPLY,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * {
      return PyNumber_InPlaceMatrixMultiply(objs[0], objs[1]);
    }}},
  {BINARY_POWER,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_Power(objs[0], objs[1], Py_None); }}},
  {BINARY_MULTIPLY,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_Multiply(objs[0], objs[1]); }}},
  {BINARY_MODULO,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * {
      if (PyUnicode_CheckExact(objs[0]) && (!PyUnicode_Check(objs[1]) || PyUnicode_CheckExact(objs[1]))) {
        return PyUnicode_Format(objs[0], objs[1]);
      } else {
        return PyNumber_Remainder(objs[0], objs[1]);
      }
    }}},
  {BINARY_ADD,
   {[](int opargs, const PyObjectArray &objs) -> bool {
      return !PyUnicode_CheckExact(objs[0]) || !PyUnicode_CheckExact(objs[1]);
    },
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_Add(objs[0], objs[1]); }}},
  {BINARY_SUBTRACT,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_Subtract(objs[0], objs[1]); }}},
  {BINARY_SUBSCR,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyObject_GetItem(objs[0], objs[1]); }}},
  {BINARY_FLOOR_DIVIDE,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_FloorDivide(objs[0], objs[1]); }}},
  {BINARY_TRUE_DIVIDE,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_TrueDivide(objs[0], objs[1]); }}},
  {INPLACE_FLOOR_DIVIDE,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_InPlaceFloorDivide(objs[0], objs[1]); }}},
  {INPLACE_TRUE_DIVIDE,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_InPlaceTrueDivide(objs[0], objs[1]); }}},
  {GET_AITER, {ByteCodeUnsupported, nullptr}},
  {GET_ANEXT, {ByteCodeUnsupported, nullptr}},
  {BEFORE_ASYNC_WITH, {ByteCodeUnsupported, nullptr}},
  {INPLACE_ADD,
   {[](int opargs, const PyObjectArray &objs) -> bool {
      return !PyUnicode_CheckExact(objs[0]) || !PyUnicode_CheckExact(objs[1]);
    },
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_InPlaceAdd(objs[0], objs[1]); }}},
  {INPLACE_SUBTRACT,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_InPlaceSubtract(objs[0], objs[1]); }}},
  {INPLACE_MULTIPLY,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_InPlaceMultiply(objs[0], objs[1]); }}},
  {INPLACE_MODULO,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_InPlaceRemainder(objs[0], objs[1]); }}},
  {STORE_SUBSCR, {ByteCodeUnsupported, nullptr}},
  {DELETE_SUBSCR, {ByteCodeUnsupported, nullptr}},
  {BINARY_LSHIFT,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_Lshift(objs[0], objs[1]); }}},
  {BINARY_RSHIFT,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_Rshift(objs[0], objs[1]); }}},
  {BINARY_AND,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_And(objs[0], objs[1]); }}},
  {BINARY_XOR,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_Xor(objs[0], objs[1]); }}},
  {BINARY_OR,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_Or(objs[0], objs[1]); }}},
  {INPLACE_POWER,
   {ByteCodeSupported,
    [](int opargs,
       const PyObjectArray &objs) -> PyObject * { return PyNumber_InPlacePower(objs[0], objs[1], Py_None); }}},
  {GET_ITER, {ByteCodeUnsupported, nullptr}},
  {GET_YIELD_FROM_ITER, {ByteCodeUnsupported, nullptr}},
  {PRINT_EXPR, {ByteCodeUnsupported, nullptr}},
  {LOAD_BUILD_CLASS, {ByteCodeUnsupported, nullptr}},
  {YIELD_FROM, {ByteCodeUnsupported, nullptr}},
  {GET_AWAITABLE, {ByteCodeUnsupported, nullptr}},
  {INPLACE_LSHIFT,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_InPlaceLshift(objs[0], objs[1]); }}},
  {INPLACE_RSHIFT,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_InPlaceRshift(objs[0], objs[1]); }}},
  {INPLACE_AND,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_InPlaceAnd(objs[0], objs[1]); }}},
  {INPLACE_XOR,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_InPlaceXor(objs[0], objs[1]); }}},
  {INPLACE_OR,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyNumber_InPlaceOr(objs[0], objs[1]); }}},
  {RETURN_VALUE, {ByteCodeUnsupported, nullptr}},
  {IMPORT_STAR, {ByteCodeUnsupported, nullptr}},
  {SETUP_ANNOTATIONS, {ByteCodeUnsupported, nullptr}},
  {YIELD_VALUE, {ByteCodeUnsupported, nullptr}},
  {POP_BLOCK, {ByteCodeUnsupported, nullptr}},
  {POP_EXCEPT, {ByteCodeUnsupported, nullptr}},
  {HAVE_ARGUMENT, {ByteCodeUnsupported, nullptr}},
  {STORE_NAME, {ByteCodeUnsupported, nullptr}},
  {DELETE_NAME, {ByteCodeUnsupported, nullptr}},
  {UNPACK_SEQUENCE, {ByteCodeUnsupported, nullptr}},
  {FOR_ITER, {ByteCodeUnsupported, nullptr}},
  {UNPACK_EX, {ByteCodeUnsupported, nullptr}},
  {STORE_ATTR, {ByteCodeUnsupported, nullptr}},
  {DELETE_ATTR, {ByteCodeUnsupported, nullptr}},
  {STORE_GLOBAL, {ByteCodeUnsupported, nullptr}},
  {DELETE_GLOBAL, {ByteCodeUnsupported, nullptr}},
  {LOAD_CONST, {ByteCodeSupported, nullptr}},
  {LOAD_NAME, {ByteCodeUnsupported, nullptr}},
  {BUILD_TUPLE,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject
                                                * {
                                                  PyObject *tup = PyTuple_New(opargs);
                                                  while (--opargs >= 0) {
                                                    Py_INCREF(objs[opargs]);
                                                    PyTuple_SET_ITEM(tup, opargs, objs[opargs]);
                                                  }
                                                  return tup;
                                                }}},
  {BUILD_LIST,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject
                                                * {
                                                  PyObject *list = PyList_New(opargs);
                                                  while (--opargs >= 0) {
                                                    Py_INCREF(objs[opargs]);
                                                    PyList_SET_ITEM(list, opargs, objs[opargs]);
                                                  }
                                                  return list;
                                                }}},
  {BUILD_SET,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject
                                                * {
                                                  PyObject *set = PySet_New(NULL);
                                                  for (int i = opargs; i > 0; i--) {
                                                    PySet_Add(set, objs[opargs - i]);
                                                  }
                                                  return set;
                                                }}},
  {BUILD_MAP,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject
                                                * {
                                                  PyObject *map = _PyDict_NewPresized((Py_ssize_t)opargs);
                                                  for (Py_ssize_t i = opargs; i > 0; i--) {
                                                    PyObject *key = objs[2 * (opargs - i)];
                                                    PyObject *value = objs[2 * (opargs - i) + 1];
                                                    PyDict_SetItem(map, key, value);
                                                  }
                                                  return map;
                                                }}},
  {LOAD_ATTR, {ByteCodeSupported, nullptr}},
  {COMPARE_OP,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject * { return RichCompare(objs[0], objs[1], opargs); }}},
  {IMPORT_NAME, {ByteCodeUnsupported, nullptr}},
  {IMPORT_FROM, {ByteCodeUnsupported, nullptr}},
  {JUMP_FORWARD, {ByteCodeUnsupported, nullptr}},
  {JUMP_IF_FALSE_OR_POP, {ByteCodeUnsupported, nullptr}},
  {JUMP_IF_TRUE_OR_POP, {ByteCodeUnsupported, nullptr}},
  {JUMP_ABSOLUTE, {ByteCodeUnsupported, nullptr}},
  {POP_JUMP_IF_FALSE, {ByteCodeUnsupported, nullptr}},
  {POP_JUMP_IF_TRUE, {ByteCodeUnsupported, nullptr}},
  {LOAD_GLOBAL, {ByteCodeSupported, nullptr}},
  {SETUP_FINALLY, {ByteCodeUnsupported, nullptr}},
  {LOAD_FAST, {ByteCodeUnsupported, nullptr}},
  {STORE_FAST, {ByteCodeUnsupported, nullptr}},
  {DELETE_FAST, {ByteCodeUnsupported, nullptr}},
  {RAISE_VARARGS, {ByteCodeUnsupported, nullptr}},
  {CALL_FUNCTION, {ByteCodeSupported, nullptr}},
  {MAKE_FUNCTION, {ByteCodeUnsupported, nullptr}},
  {BUILD_SLICE,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject
                                                * {
                                                  PyObject *start, *stop, *step;
                                                  if (opargs == 3)
                                                    step = objs[2];
                                                  else
                                                    step = nullptr;
                                                  stop = objs[1];
                                                  start = objs[0];
                                                  return PySlice_New(start, stop, step);
                                                }}},
  {LOAD_CLOSURE, {ByteCodeSupported, nullptr}},
  {LOAD_DEREF, {ByteCodeSupported, nullptr}},
  {STORE_DEREF, {ByteCodeUnsupported, nullptr}},
  {DELETE_DEREF, {ByteCodeUnsupported, nullptr}},
  {CALL_FUNCTION_KW, {ByteCodeSupported, nullptr}},
  {CALL_FUNCTION_EX, {ByteCodeSupported, nullptr}},
  {SETUP_WITH, {ByteCodeUnsupported, nullptr}},
  {EXTENDED_ARG, {ByteCodeUnsupported, nullptr}},
  {LIST_APPEND,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject
                                                * {
                                                  PyList_Append(objs[0], objs[1]);
                                                  Py_INCREF(objs[0]);
                                                  return objs[0];
                                                }}},
  {SET_ADD,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject
                                                * {
                                                  PySet_Add(objs[0], objs[1]);
                                                  Py_INCREF(objs[0]);
                                                  return objs[0];
                                                }}},
  {MAP_ADD,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject
                                                * {
                                                  PyDict_SetItem(objs[0], objs[2], objs[1]);
                                                  Py_INCREF(objs[0]);
                                                  return objs[0];
                                                }}},
  {LOAD_CLASSDEREF, {ByteCodeUnsupported, nullptr}},
  {SETUP_ASYNC_WITH, {ByteCodeUnsupported, nullptr}},
  {FORMAT_VALUE, {ByteCodeUnsupported, nullptr}},
  {BUILD_CONST_KEY_MAP,
   {[](int opargs, const PyObjectArray &objs) -> bool {
      return PyTuple_CheckExact(objs[opargs]) && PyTuple_GET_SIZE(objs[opargs]) == (Py_ssize_t)opargs;
    },
    [](int opargs, const PyObjectArray &objs) -> PyObject * {
      PyObject *keys = objs[opargs];
      PyObject *map = _PyDict_NewPresized((Py_ssize_t)opargs);
      for (Py_ssize_t i = opargs; i > 0; i--) {
        PyObject *key = PyTuple_GET_ITEM(keys, opargs - i);
        PyObject *value = objs[opargs - i];
        PyDict_SetItem(map, key, value);
      }
      return map;
    }}},
  {BUILD_STRING,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject
                                                * {
                                                  PyObject *empty = PyUnicode_New(0, 0);
                                                  PyObject *str = _PyUnicode_JoinArray(empty, objs.data(), opargs);
                                                  Py_DECREF(empty);
                                                  return str;
                                                }}},
  {LOAD_METHOD, {ByteCodeUnsupported, nullptr}},
  {CALL_METHOD, {ByteCodeUnsupported, nullptr}},
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION == 9)
  {ROT_FOUR, {ByteCodeUnsupported, nullptr}},
  {RERAISE, {ByteCodeUnsupported, nullptr}},
  {WITH_EXCEPT_START, {ByteCodeUnsupported, nullptr}},
  {END_ASYNC_FOR, {ByteCodeUnsupported, nullptr}},
  {LOAD_ASSERTION_ERROR, {ByteCodeUnsupported, nullptr}},
  {LIST_TO_TUPLE,
   {ByteCodeSupported, [](int opargs, const PyObjectArray &objs) -> PyObject * { return PyList_AsTuple(objs[0]); }}},
  {IS_OP,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject
                                                * {
                                                  auto ret = (objs[0] == objs[1]) ^ opargs ? Py_True : Py_False;
                                                  Py_INCREF(ret);
                                                  return ret;
                                                }}},
  {CONTAINS_OP,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject
                                                * {
                                                  auto ret = (PySequence_Contains(objs[1], objs[0]) ^ opargs)
                                                               ? Py_True
                                                               : Py_False;
                                                  Py_INCREF(ret);
                                                  return ret;
                                                }}},
  {JUMP_IF_NOT_EXC_MATCH, {ByteCodeUnsupported, nullptr}},
  {LIST_EXTEND,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject
                                                * {
                                                  _PyList_Extend(reinterpret_cast<PyListObject *>(objs[0]), objs[1]);
                                                  Py_INCREF(objs[0]);
                                                  return objs[0];
                                                }}},
  {SET_UPDATE,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject
                                                * {
                                                  _PySet_Update(objs[0], objs[1]);
                                                  Py_INCREF(objs[0]);
                                                  return objs[0];
                                                }}},
  {DICT_MERGE,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject
                                                * {
                                                  _PyDict_MergeEx(objs[0], objs[1], 2);
                                                  return objs[0];
                                                }}},
  {DICT_UPDATE,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject
                                                * {
                                                  PyDict_Update(objs[0], objs[1]);
                                                  return objs[0];
                                                }}},
#endif
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION == 7)
  {BREAK_LOOP, {ByteCodeUnsupported, nullptr}},
  {WITH_CLEANUP_START, {ByteCodeUnsupported, nullptr}},
  {WITH_CLEANUP_FINISH, {ByteCodeUnsupported, nullptr}},
  {END_FINALLY, {ByteCodeUnsupported, nullptr}},
  {CONTINUE_LOOP, {ByteCodeUnsupported, nullptr}},
  {SETUP_LOOP, {ByteCodeUnsupported, nullptr}},
  {SETUP_EXCEPT, {ByteCodeUnsupported, nullptr}},
  {BUILD_LIST_UNPACK,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject
                                                * {
                                                  PyObject *sum = PyList_New(0);
                                                  for (int i = opargs; i > 0; i--) {
                                                    auto none_val = _PyList_Extend(
                                                      reinterpret_cast<PyListObject *>(sum), objs[opargs - i]);
                                                    Py_DECREF(none_val);
                                                  }
                                                  return sum;
                                                }}},
  {BUILD_MAP_UNPACK,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject
                                                * {
                                                  PyObject *sum = PyDict_New();
                                                  for (int i = opargs; i > 0; i--) {
                                                    PyDict_Update(sum, objs[opargs - i]);
                                                  }
                                                  return sum;
                                                }}},
  {BUILD_MAP_UNPACK_WITH_CALL,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject
                                                * {
                                                  PyObject *sum = PyDict_New();
                                                  for (int i = opargs; i > 0; i--) {
                                                    _PyDict_MergeEx(sum, objs[opargs - i], 2);
                                                  }
                                                  return sum;
                                                }}},
  {BUILD_TUPLE_UNPACK,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject
                                                * {
                                                  PyObject *sum = PyList_New(0);
                                                  for (int i = opargs; i > 0; i--) {
                                                    auto none_val = _PyList_Extend(
                                                      reinterpret_cast<PyListObject *>(sum), objs[opargs - i]);
                                                    Py_DECREF(none_val);
                                                  }
                                                  auto ret = PyList_AsTuple(sum);
                                                  Py_DECREF(sum);
                                                  return ret;
                                                }}},
  {BUILD_SET_UNPACK,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject
                                                * {
                                                  PyObject *sum = PySet_New(NULL);
                                                  for (int i = opargs; i > 0; i--) {
                                                    _PySet_Update(sum, objs[opargs - i]);
                                                  }
                                                  return sum;
                                                }}},
  {BUILD_TUPLE_UNPACK_WITH_CALL,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs) -> PyObject
                                                * {
                                                  PyObject *sum = PyList_New(0);
                                                  for (int i = opargs; i > 0; i--) {
                                                    auto none_val = _PyList_Extend(
                                                      reinterpret_cast<PyListObject *>(sum), objs[opargs - i]);
                                                    Py_DECREF(none_val);
                                                  }
                                                  auto ret = PyList_AsTuple(sum);
                                                  Py_DECREF(sum);
                                                  return ret;
                                                }}},
  {EXCEPT_HANDLER, {ByteCodeUnsupported, nullptr}},
#endif
};

OpTrace::OpTrace(PyObject *obj, int opcode, int opargs, TraceVector params, std::string name)
    : Trace(obj, nullptr), opcode_(opcode), opargs_(opargs), params_(params), name_(name) {
  curType_ = TraceType::Operation;
}

PyObject *OpTrace::Retrieve(PTraceContext context) {
  PyObject *ret = NULL;
  std::vector<PyObject *> params;
  auto iter = std::find_if(params_.begin(), params_.end(), [&params, &context](const TracePtr &p) {
    auto param = p->Retrieve(context);
    if (param == nullptr) {
      return true;
    }
    if (py::isinstance<mindspore::tensor::Tensor>(param)) {
      mindspore::tensor::TensorPtr tensor_ptr = py::cast<mindspore::tensor::TensorPtr>(param);
      tensor_ptr->data_sync(true);
    }
    params.push_back(param);
    return params.back() == nullptr;
  });
  if (iter != params_.end()) {
    MS_LOG(DEBUG) << "Guard Check Retrieve fail for " + (*iter)->ToString();
    std::for_each(params.begin(), params.end(), [](PyObject *p) { Py_XDECREF(p); });
    return nullptr;
  }
  if (kBytecodeExecuter.find(opcode_) != kBytecodeExecuter.end() && kBytecodeExecuter[opcode_].first(opargs_, params) &&
      kBytecodeExecuter[opcode_].second != nullptr) {
    ret = kBytecodeExecuter[opcode_].second(opargs_, params);
  } else {
    switch (opcode_) {
      case LOAD_ATTR:
        MS_EXCEPTION_IF_CHECK_FAIL(name_.size(), "check trace");
        ret = PyObject_GetAttrString(params[0], name_.c_str());
        break;
      case CALL_FUNCTION:
      case CALL_FUNCTION_EX:
      case CALL_FUNCTION_KW:
        ret = DoCall(params, opcode_, name_);
        break;
      /* fall-through */
      default:
        break;
    }
  }
  for (auto p : params) {
    Py_DECREF(p);
  }
  return ret;
}

std::string OpTrace::ToString() {
  std::string ret = "operation ";
  ret += Utils::GetOpName(opcode_) + "(";
  if (name_.size() != 0) {
    ret += std::string("name:") + name_;
    if (params_.size() > 0) {
      ret += ",";
    }
  }
  if (params_.size() > 0) {
    for (auto t : params_) {
      ret += t->ToString() + ",";
    }
    ret = ret.substr(0, ret.size() - 1);
  }
  return ret + ")";
}

bool OpTrace::operator==(const Trace &trace) {
  bool ret = false;
  if (Trace::operator==(trace)) {
    const OpTrace &t = (const OpTrace &)trace;
    ret = opcode_ == t.opcode_ && opargs_ == t.opargs_ && name_ == t.name_ && params_.size() == t.params_.size();
    if (ret) {
      for (size_t i = 0; i < params_.size(); i++) {
        if (*(params_[i]) == *(t.params_[i])) {
          continue;
        } else {
          ret = false;
          break;
        }
      }
    }
  }
  return ret;
}

TracePtr CreateOpTrace(PyObject *obj, int opcode, int opargs, TraceVector params, std::string module_name,
                       std::string name, bool strict, bool print) {
  std::vector<PyObject *> vparams;
  for (auto trace : params) {
    if (trace == nullptr) {
      return nullptr;
    } else if (trace->GetTraceType() == TraceType::Unsupported) {
      return std::make_shared<UnsupportedTrace>(obj, params, opcode, opargs);
    } else {
      vparams.push_back(trace->GetObject());
    }
  }
  if (kBytecodeExecuter.find(opcode) == kBytecodeExecuter.end() || !kBytecodeExecuter[opcode].first(opargs, vparams)) {
    if (print) {
      GRAPH_JIT_LOG_F("Unsupported bytecode %d args %d!\n", opcode, opargs);
    } else {
      MS_LOG(DEBUG) << "Unsupported bytecode " << opcode << " args " << opargs << "!";
    }
    if (strict) {
      return nullptr;
    } else {
      return std::make_shared<UnsupportedTrace>(obj, params, opcode, opargs);
    }
  }
  switch (opcode) {
    case LOAD_CLOSURE:
      return std::make_shared<RootTrace>(obj, TraceType::Closure, opargs, name, module_name);
    case LOAD_DEREF:
      return std::make_shared<RootTrace>(obj, TraceType::Deref, opargs, name, module_name);
    case LOAD_GLOBAL:
      return std::make_shared<RootTrace>(obj, TraceType::Global, opargs, name, module_name);
    case LOAD_CONST:
      return std::make_shared<ConstTrace>(obj, opargs);
    case CALL_FUNCTION:
    case CALL_FUNCTION_EX:
    case CALL_FUNCTION_KW:
      if (params.size() < 1 || !SupportCall(params[0]->GetObject(), name)) {
        if (strict) {
          return nullptr;
        } else {
          return std::make_shared<UnsupportedTrace>(obj, params, opcode, opargs);
        }
      }
    default:
      break;
  }
  return std::make_shared<OpTrace>(obj, opcode, opargs, params, name);
}

CustomizedTrace::CustomizedTrace(PyObject *obj, RetrieveFunc rfunc, ToStringFunc sfunc)
    : Trace(obj, nullptr), retrieve_(rfunc), tostring_(sfunc) {
  curType_ = TraceType::Customized;
}

PyObject *CustomizedTrace::Retrieve(PTraceContext context) { return retrieve_(context); }

std::string CustomizedTrace::ToString() { return tostring_(); }

UnsupportedTrace::UnsupportedTrace(PyObject *obj, TraceVector params, int op, int arg)
    : Trace(obj, nullptr), params_(params), op_(op), arg_(arg) {
  curType_ = TraceType::Unsupported;
}

PyObject *UnsupportedTrace::Retrieve(PTraceContext context) {
  std::vector<PyObject *> params;
  bool fail = false;
  for (auto p : params_) {
    auto obj = p->Retrieve(context);
    params.push_back(obj);
    if (p->GetTraceType() != TraceType::Unsupported) {
      // compare obj with original obj in trace for inputs of unsupported trace
      auto orig = p->GetObject();
      if (!IsPyObjectEqual(obj, orig)) {
        fail = true;
        break;
      }
    }
    if (params.back() == nullptr) {
      MS_LOG(DEBUG) << "Guard Check Retrieve fail for " + p->ToString();
      fail = true;
      break;
    }
  }
  for (auto p : params) {
    Py_XDECREF(p);
  }
  if (fail) {
    return nullptr;
  } else {
    return obj_;
  }
}

std::string UnsupportedTrace::ToString() {
  std::string ret = "unsupported(";
  if (params_.size() > 0) {
    for (auto t : params_) {
      ret += t->ToString() + ",";
    }
    ret = ret.substr(0, ret.size() - 1);
  }
  return ret + ")";
}

TraceVector UnsupportedTrace::GetParams() { return params_; }

void UnsupportedTrace::Detach() {
  Trace::Detach();
  for (auto t : params_) {
    t->Detach();
  }
}

PyObject *GetObjectFromTrace(const PyFrameObject *frame, TracePtr trace) {
  TraceContext context = {frame->f_globals, frame->f_builtins, frame->f_locals, frame->f_localsplus, frame->f_code};
  if (trace != NULL) {
    return trace->Retrieve(&context);
  } else {
    return NULL;
  }
}

}  // namespace graph
}  // namespace jit
}  // namespace mindspore
