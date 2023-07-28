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
#include "pipeline/jit/graph_jit/graph_guard/trace.h"
#include <map>
#include <vector>
#include "pybind11/pybind11.h"
#include "pybind_api/ir/primitive_py.h"
#include "include/common/utils/convert_utils_py.h"
#include "pipeline/jit/graph_jit/graph_guard/infer.h"
#include "pipeline/jit/graph_jit/utils/utils.h"
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

static std::map<int, std::string> g_mapOp = {
  {LOAD_GLOBAL, "LOAD_GLOBAL"},     {LOAD_CONST, "LOAD_CONST"},       {LOAD_ATTR, "LOAD_ATTR"},
  {COMPARE_OP, "COMPARE_OP"},       {CONTAINS_OP, "CONTAINS_OP"},     {IS_OP, "IS_OP"},
  {BUILD_TUPLE, "BUILD_TUPLE"},     {CALL_FUNCTION, "CALL_FUNCTION"}, {BINARY_SUBSCR, "BINARY_SUBSCR"},
  {LIST_TO_TUPLE, "LIST_TO_TUPLE"}, {LIST_APPEND, "LIST_APPEND"},     {LIST_EXTEND, "LIST_EXTEND"},
  {BUILD_LIST, "BUILD_LIST"},
};

OpTrace::OpTrace(PyObject *obj, int opcode, int opargs, TraceVector params, std::string name)
    : Trace(obj, nullptr), opcode_(opcode), opargs_(opargs), params_(params), name_(name) {
  curType_ = TraceType::Operation;
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

static PyObject *DoCall(const std::vector<PyObject *> &params, int op) {
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
  if (!support_create_primitive(params[0])) {
    // current only for builtin func
    MS_EXCEPTION_IF_CHECK_FAIL(check_builtin_cfunc(py::cast<py::object>(params[0])),
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

PyObject *OpTrace::Retrieve(PTraceContext context) {
  PyObject *ret = NULL;
  std::vector<PyObject *> params;
  auto iter = std::find_if(params_.begin(), params_.end(), [&params, &context](const TracePtr &p) {
    params.push_back(p->Retrieve(context));
    return params.back() == nullptr;
  });
  if (iter != params_.end()) {
    MS_LOG(DEBUG) << "Guard Check Retrieve fail for " + (*iter)->ToString();
    std::for_each(params.begin(), params.end(), [](PyObject *p) { Py_XDECREF(p); });
    return nullptr;
  }
  switch (opcode_) {
    case LOAD_ATTR:
      MS_EXCEPTION_IF_CHECK_FAIL(name_.size(), "check trace");
      ret = PyObject_GetAttrString(params[0], name_.c_str());
      break;
    case COMPARE_OP:
      ret = RichCompare(params[0], params[1], opargs_);
      break;
    case IS_OP:
      ret = ((params[0] == params[1]) ^ opargs_) ? Py_True : Py_False;
      Py_INCREF(ret);
      break;
    case CONTAINS_OP:
      ret = (PySequence_Contains(params[1], params[0]) ^ opargs_) ? Py_True : Py_False;
      if (ret == nullptr) {
        PyErr_Clear();
      } else {
        Py_INCREF(ret);
      }
      break;
    case BINARY_SUBSCR:
      ret = PyObject_GetItem(params[0], params[1]);
      break;
    case CALL_FUNCTION:
      ret = DoCall(params, opcode_);
      break;
    case BUILD_TUPLE:
    case BUILD_LIST:
      ret = PyList_New(opargs_);
      for (Py_ssize_t i = 0; i < opargs_; ++i) {
        Py_INCREF(params[i]);
        PyList_SetItem(ret, i, params[i]);
      }
      if (opcode_ == BUILD_LIST) {
        break;
      }
      Py_SETREF(params[0], ret);
    /* fall-through */
    case LIST_TO_TUPLE:
      ret = PyList_AsTuple(params[0]);
      break;
    case LIST_APPEND:
      PyList_Append(params[0], params[1]);
      ret = params[0];
      Py_INCREF(ret);
      break;
    case LIST_EXTEND:
      _PyList_Extend(reinterpret_cast<PyListObject *>(params[0]), params[1]);
      ret = params[0];
      Py_INCREF(ret);
      break;
    default:
      break;
  }
  for (auto p : params) {
    Py_DECREF(p);
  }
  return ret;
}

std::string OpTrace::ToString() {
  std::string ret = "operation ";
  ret += g_mapOp[opcode_] + "(";
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

bool SupportCall(PyObject *func, const std::string &name) {
  return support_infer_primitive(func) || support_create_primitive(func) || name == "len" || name == "isinstance";
}

TracePtr CreateOpTrace(PyObject *obj, int opcode, int opargs, TraceVector params, std::string module_name,
                       std::string name, bool print) {
  if (g_mapOp.find(opcode) == g_mapOp.end()) {
    if (print) {
      GRAPH_JIT_LOG_F("Unsupported bytecode %d args %d!\n", opcode, opargs);
    } else {
      MS_LOG(DEBUG) << "Unsupported bytecode " << opcode << " args " << opargs << "!";
    }
    return nullptr;
  }
  switch (opcode) {
    case LOAD_GLOBAL:
      return std::make_shared<RootTrace>(obj, TraceType::Global, opargs, name, module_name);
    case LOAD_CONST:
      return std::make_shared<ConstTrace>(obj, opargs);
    case CALL_FUNCTION:
      if (params.size() < 1 || !SupportCall(params[0]->GetObject(), name)) {
        return nullptr;
      }
    default:
      break;
  }
  return std::make_shared<OpTrace>(obj, opcode, opargs, params, name);
}

CustomizedTrace::CustomizedTrace(PyObject *obj, RetrieveFunc rfunc, ToStringFunc sfunc)
    : Trace(obj, nullptr), retrieve_(rfunc), tostring_(sfunc) {}

PyObject *CustomizedTrace::Retrieve(PTraceContext context) { return retrieve_(context); }

std::string CustomizedTrace::ToString() { return tostring_(); }

PyObject *GetObjectFromTrace(PyFrameObject *frame, TracePtr trace) {
  TraceContext context = {frame->f_globals, frame->f_builtins, frame->f_locals, frame->f_localsplus, frame->f_code};
  if (trace != NULL) {
    PyObject *obj = trace->Retrieve(&context);
    if (obj == NULL || obj == Py_None) {
      return obj;
    }
    py::object py_obj = py::reinterpret_borrow<py::object>(obj);
    if (IsStubTensor(py_obj)) {
      py_obj = python_adapter::CallPyObjMethod(py_obj, "stub_sync");
      obj = py_obj.ptr();
      Py_INCREF(obj);
    }
    return obj;
  } else {
    return NULL;
  }
}

}  // namespace graph
}  // namespace jit
}  // namespace mindspore
