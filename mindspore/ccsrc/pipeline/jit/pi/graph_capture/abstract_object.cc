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
#include "pipeline/jit/pi/graph_capture/abstract_object.h"
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include "utils/log_adapter.h"
#include "pipeline/jit/pi/utils/utils.h"
#include "pipeline/jit/pi/pydef.h"
#include "pipeline/jit/pi/graph_guard/infer.h"
#include "pipeline/jit/pi/graph_compiler/utils.h"
#include "pipeline/jit/pi/graph_compiler/pi_ir/ctrl_flow.h"
#include "pipeline/jit/pi/graph_compiler/pi_ir/custom_nodes.h"
#include "pipeline/jit/pi/graph_compiler/pi_ir/operation.h"
#include "pipeline/jit/pi/graph_compiler/pi_ir/value.h"
#include "pipeline/jit/ps/action.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "mindspore/core/ops/math_ops.h"
#include "include/common/utils/convert_utils_py.h"

namespace mindspore {
namespace pijit {
#define FIND_MAP_CACHE(map, target) \
  do {                              \
    auto iter = (map).find(target); \
    if (iter != (map).end()) {      \
      return iter->second;          \
    }                               \
  } while (0)

#ifdef DEBUG
#define CHECK_PYTHON_EXCEPTION(check_res)       \
  if (PyErr_Occurred()) {                       \
    MS_LOG(DEBUG) << "has an python exception"; \
    MS_ASSERT((check_res) == nullptr);          \
    PyErr_Print();                              \
    PyErr_Clear();                              \
  }
#else
#define CHECK_PYTHON_EXCEPTION(check_res) PyErr_Clear()
#endif

// mindspore graph can accept these value
static const std::set<AObject::Type> kMsSupportedType = {
  AObject::kTypeInt,  AObject::kTypeBool,   AObject::kTypeFloat,
  AObject::kTypeNone, AObject::kTypeString, AObject::kTypeTensor,
  AObject::kTypeFuncGraphOut,
};

MemPool<AbstractObjectBase> AbstractObjectBase::aobject_mem_pool_(__FILE__, __LINE__, "AObject");

// exact equal check
static const std::unordered_map<PyTypeObject *, AObject::Type> exact_type_map = {
  {&PyFunction_Type, AObject::kTypeFunction},
  {&PyMethod_Type, AObject::kTypeBoundMethod},
  {&PyCode_Type, AObject::kTypeCodeObject},
  {&PySlice_Type, AObject::kTypeSlice},
  {&PySet_Type, AObject::kTypeSet},
  {&PyFrozenSet_Type, AObject::kTypeSet},
  {&PyBool_Type, AObject::kTypeBool},
  {&PyFloat_Type, AObject::kTypeFloat},
  {&PyLong_Type, AObject::kTypeInt},
  {&PyList_Type, AObject::kTypeList},
  {&PyTuple_Type, AObject::kTypeTuple},
  {&PyDict_Type, AObject::kTypeDict},
  {&PyDictValues_Type, AObject::kTypeDictValues},
  {&PyDictKeys_Type, AObject::kTypeDictKeys},
  {&PyDictItems_Type, AObject::kTypeDictItems},
  {&PyType_Type, AObject::kTypeType},
  {&PyUnicode_Type, AObject::kTypeString},
  {&PyModule_Type, AObject::kTypeModule},
  {&PyCFunction_Type, AObject::kTypeCFunction},
  {nullptr, AObject::kTypeAnyValue},
};

// shouldn't add nullptr to this map
static const std::unordered_map<PyObject *, AObject::Type> const_object_type_map = {
  {Py_Ellipsis, AObject::kTypeEllipsis},
  {Py_None, AObject::kTypeNone},
  {Py_True, AObject::kTypeBool},
  {Py_False, AObject::kTypeBool},
};

static const std::vector<std::pair<PyTypeObject *, AObject::Type>> sub_type_map = {
  {&PyModule_Type, AObject::kTypeModule}, {&PyCFunction_Type, AObject::kTypeCFunction}};

static const int fast_type_mask = Py_TPFLAGS_LONG_SUBCLASS | Py_TPFLAGS_LIST_SUBCLASS | Py_TPFLAGS_TUPLE_SUBCLASS |
                                  Py_TPFLAGS_UNICODE_SUBCLASS | Py_TPFLAGS_DICT_SUBCLASS | Py_TPFLAGS_TYPE_SUBCLASS;

const char *AbstractObjectBase::GetTypeDesc(AObject::Type type) {
  switch (type) {
#define ABSTRACT_TYPE_DEF(unit) \
  case AObject::kType##unit:    \
    return "kType" #unit;
#include "abstract_type_kind.def"
#undef ABSTRACT_TYPE_DEF
  }
  return nullptr;
}

bool AbstractObjectBase::IsMindSporeSupportedType() {
  return kMsSupportedType.find(GetType()) != kMsSupportedType.end();
}

std::string AbstractObjectBase::ToString(PyObject *op) {
  if (op == nullptr) {
    return "<NULL>";
  }
  ReprRecursionScope scope(op);
  if (scope.ReEnter()) {
    return "...";
  }

  py::object obj = py::cast<py::object>(op);
  AObject::Type t = AObject::GetPyType(op);
  std::stringstream s;
  s << std::string(py::str(reinterpret_cast<PyObject *>(Py_TYPE(op)))) << "{ ";
  switch (t) {
    case AObject::kTypeTensor:
    case AObject::kTypeStubTensor: {
      s << std::string(py::str(obj.attr("shape"))) << ", " << std::string(py::str(obj.attr("dtype")));
      break;
    }
    case AObject::kTypeBoundMethod: {
      s << std::string(py::str(PyMethod_GET_FUNCTION(op))) << " at " << ToString(PyMethod_GET_SELF(op));
      break;
    }
    case AObject::kTypeNNCellList:
    case AObject::kTypeList:
    case AObject::kTypeTuple: {
      s << (t == AObject::kTypeTuple ? "( " : "[ ");
      for (auto i : py::iter(obj)) {
        s << ToString(i.ptr()) << ", ";
      }
      s.seekp(-2, s.cur);
      s << (t == AObject::kTypeTuple ? " )" : " ]");
      break;
    }
    case AObject::kTypeDict: {
      PyObject *key;
      PyObject *val;
      Py_ssize_t pos = 0;
      s << "{ ";
      while (PyDict_Next(op, &pos, &key, &val)) {
        s << ToString(key) << ":" << ToString(val) << ", ";
      }
      s.seekp(-2, s.cur);
      s << " }";
      break;
    }
    case AObject::kTypeAnyValue:
    case AObject::kTypeCell: {
      s << " at " << op;
      break;
    }
    default:
      s << std::string(py::str(obj));
      break;
  }
  s << " }";
  return s.str();
}

std::string AbstractObjectBase::ToString() const {
  std::string s = " ";
#define ABSTRACT_MS_FLAG_DEF(unit, bit) s += ((ms_flag_ & kMsFlag##unit) ? #unit "|" : "");
#include "abstract_ms_flag.def"
#undef ABSTRACT_MS_FLAG_DEF
  if (s.back() == '|') {
    s.pop_back();
  }
  if (type_object_ != nullptr) {
    s += std::string(py::str(reinterpret_cast<PyObject *>(type_object_)));
  }
  return GetTypeDesc(GetType()) + s;
}

AbstractObjectBase::Type AbstractObjectBase::GetPyType(PyTypeObject *tp) {
  FIND_MAP_CACHE(exact_type_map, tp);
  // fast sub type check
  // __builtin_clz(tp->tp_flags & fast_type_mask), or std::countl_zero
  /**
   * sub-class int, float, list, tuple, str, is mindspore unsupported
   */
  switch (tp->tp_flags & fast_type_mask) {
    case Py_TPFLAGS_LONG_SUBCLASS:
    case Py_TPFLAGS_LIST_SUBCLASS:
    case Py_TPFLAGS_TUPLE_SUBCLASS:
    case Py_TPFLAGS_UNICODE_SUBCLASS:
    case Py_TPFLAGS_DICT_SUBCLASS:
      return kTypeAnyValue;
    case Py_TPFLAGS_TYPE_SUBCLASS:
      return kTypeType;
    default:
      break;
  }
  // sub type check
  for (auto &i : sub_type_map) {
    if (PyType_IsSubtype(tp, i.first)) {
      return i.second;
    }
  }
  return GetMsType(tp);
}

AbstractObjectBase::Type AbstractObjectBase::GetPyType(PyObject *o) {
  if (o == nullptr) {
    return kTypeAnyValue;
  }
  FIND_MAP_CACHE(const_object_type_map, o);
  if (PyLong_Check(o)) {
    return (Py_ABS(Py_SIZE(o)) > 2) ? kTypeAnyValue : kTypeInt;
  }
  return GetPyType(Py_TYPE(o));
}

AbstractObjectBase::Type AbstractObjectBase::GetMsType(PyTypeObject *tp) {
  static const std::vector<std::pair<bool (*)(PyTypeObject *), AObject::Type>> match_func = {
    {IsStubTensorType<true>, kTypeStubTensor}, {IsTensorType<true>, kTypeTensor},
    {IsCellListType<false>, kTypeNNCellList},  {IsCellType<true>, kTypeCell},
    {IsPrimitiveType<true>, kTypePrimitive},   {IsMetaFuncGraphType<true>, kTypeMetaFuncGraph},
    {IsMSDTypeType<true>, kTypeMSDType},
  };
  if (tp == nullptr) {
    return kTypeAnyValue;
  }
  for (auto i : match_func) {
    if (i.first(tp)) {
      return i.second;
    }
  }
  return kTypeAnyValue;
}

AObject *AbstractObjectBase::MakeAObject(AObject::Type type, PyTypeObject *tp, PyObject *o, RecMap *m) {
  MS_EXCEPTION_IF_CHECK_FAIL(tp == nullptr || o == nullptr || Py_TYPE(o) == tp, "check type match value");
  py::object h = py::cast<py::object>(o);
  AObject *res;
  switch (type) {
    case kTypeStubTensor:
    case kTypeTensor:
      res = aobject_mem_pool_.New<AbstractTensor>(h, type == kTypeStubTensor);
      break;
    case kTypeType:
      res = aobject_mem_pool_.New<AbstractType>(h);
      break;
    case kTypeString:
      res = aobject_mem_pool_.New<AbstractSequence>(kTypeString, h);
      break;
    case kTypeNNCellList:
      res = aobject_mem_pool_.New<AbstractSequence>(kTypeNNCellList, h);
      break;
    case kTypeList:
      res = aobject_mem_pool_.New<AbstractList>(h, m);
      break;
    case kTypeTuple:
      res = aobject_mem_pool_.New<AbstractTuple>(h, m);
      break;
    case kTypeDict:
      res = aobject_mem_pool_.New<AbstractDict>(h, m);
      break;
    case kTypeAnyValue:
      if (tp == nullptr) {
        res = aobject_mem_pool_.New<AbstractObjectBase>(kTypeAnyValue);
        break;
      }
    /* fall-through */
    default:
      // known type
      res = aobject_mem_pool_.New<AbstractObject>(type, h);
      break;
  }
  res->SetTypeObject(o == nullptr ? tp : Py_TYPE(o));
  return res;
}

AObject *AbstractObjectBase::MakeFunction(const std::vector<AObject *> &args, const py::object &globals, int oparg) {
  std::vector<py::object> pyarg;
  std::transform(args.begin(), args.end(), std::back_inserter(pyarg), [](AObject *i) { return i->GetPyObject(); });
  auto iter = pyarg.end() - 1;
  PyObject *qualname = (*iter--).ptr();
  PyObject *code = (*iter--).ptr();
  py::object f_handle = py::reinterpret_steal<py::object>(PyFunction_NewWithQualName(code, globals.ptr(), qualname));
  PyFunctionObject *func = reinterpret_cast<PyFunctionObject *>(f_handle.ptr());
  MS_EXCEPTION_IF_CHECK_FAIL(func, "MAKE_FUNCTION failed");
  if (oparg & 0x08) {
    func->func_closure = (*iter--).inc_ref().ptr();
    Py_ssize_t nfrees = PyTuple_GET_SIZE(reinterpret_cast<PyCodeObject *>(code)->co_freevars);
    bool is_valid = func->func_closure && nfrees == PyTuple_GET_SIZE(func->func_closure);
    MS_EXCEPTION_IF_CHECK_FAIL(is_valid, "must be has python objects, and it is tuple of cell objects");
  }
  if (oparg & 0x04) {
    func->func_annotations = (*iter--).inc_ref().ptr();
    MS_EXCEPTION_IF_CHECK_FAIL(func->func_annotations, "must be has python objects, and it is const key map");
  }
  if (oparg & 0x02) {
    func->func_kwdefaults = (*iter--).inc_ref().ptr();
    MS_EXCEPTION_IF_CHECK_FAIL(func->func_kwdefaults, "must be has python objects, and it is const key map");
  }
  if (oparg & 0x01) {
    func->func_defaults = (*iter--).inc_ref().ptr();
    MS_EXCEPTION_IF_CHECK_FAIL(func->func_defaults, "must be has python objects, and it is const tuple");
  }
  AObject *res = AObject::Convert(f_handle);
  return res;
}

py::object AbstractObjectBase::BuildOperations(const std::vector<py::object> &args, int opcode) {
  PyObject *res = nullptr;
  PyObject **tmp;
  std::vector<PyObject *> arr;
  switch (opcode) {
    case BUILD_SLICE:
      res = PySlice_New(args[0].ptr(), args[1].ptr(), args.size() > 2 ? args[2].ptr() : nullptr);
      break;
    case BUILD_STRING:
      std::transform(args.begin(), args.end(), std::back_inserter(arr), [](const py::object &o) { return o.ptr(); });
      res = _PyUnicode_JoinArray(py::str().ptr(), arr.data(), arr.size());
      break;
    case BUILD_SET:
      res = PySet_New(nullptr);
      (void)std::find_if(args.begin(), args.end(), [&res](const py::object &i) { return PySet_Add(res, i.ptr()); });
      break;
    case BUILD_LIST:
      res = PyList_New(args.size());
      tmp = &PyList_GET_ITEM(res, 0);
      std::for_each(args.begin(), args.end(), [&tmp](const py::object &i) { return *(tmp++) = i.inc_ref().ptr(); });
      break;
    case BUILD_TUPLE:
      res = PyTuple_New(args.size());
      tmp = &PyTuple_GET_ITEM(res, 0);
      std::for_each(args.begin(), args.end(), [&tmp](const py::object &i) { return *(tmp++) = i.inc_ref().ptr(); });
      break;
    case BUILD_CONST_KEY_MAP:
      res = PyDict_New();
      // must be tuple, here has a cast check
      tmp = &PyTuple_GET_ITEM(args.back().ptr(), 0);
      (void)std::find_if(args.begin(), args.end() - 1, [&res, &tmp](const py::object &i) {
        return PyDict_SetItem(res, *(tmp++), i.ptr());  // break if err_ocurred
      });
      break;
    case BUILD_MAP:
      res = PyDict_New();
      for (size_t i = 0; !PyErr_Occurred() && i < args.size(); i += 2) {
        PyDict_SetItem(res, args[i].ptr(), args[i + 1].ptr());
      }
      break;
    default:
      break;
  }
  if (PyErr_Occurred()) {
    Py_XDECREF(res);
    MS_LOG(DEBUG) << "build operation failed: " << Utils::GetOpName(opcode);
    PyErr_Clear();
    res = nullptr;
  }
  return py::reinterpret_steal<py::object>(res);
}

AObject *AbstractObjectBase::BuildOperations(const std::vector<AObject *> &inputs, int opcode) {
  bool build_pyobject = true;
  std::vector<py::object> args;
  for (auto i = inputs.begin(); i != inputs.end() && build_pyobject; ++i) {
    args.push_back(((*i) != nullptr) ? (*i)->GetPyObject() : py::object());
    build_pyobject &= args.back().ptr() != nullptr;
  }
  if (build_pyobject) {
    return Convert(BuildOperations(args, opcode));
  }

  AObject *res = nullptr;
  PyObject *keys;
  bool err = false;
  switch (opcode) {
    case BUILD_LIST:
    case BUILD_TUPLE:
      res = MakeAObject(opcode == BUILD_LIST ? kTypeList : kTypeTuple);
      static_cast<AbstractTuple *>(res)->Update(inputs);
      break;
    case BUILD_CONST_KEY_MAP:
      res = MakeAObject(kTypeDict);
      keys = inputs.back()->GetPyObject().ptr();
      err = static_cast<Py_ssize_t>(inputs.size() - 1) != PyTuple_GET_SIZE(keys);
      for (Py_ssize_t i = inputs.size() - 2; !err && i >= 0; --i) {
        err = !static_cast<AbstractDict *>(res)->MapAdd(Convert(PyTuple_GET_ITEM(keys, i)), inputs[i]);
      }
      break;
    case BUILD_MAP:
      res = MakeAObject(kTypeDict);
      for (size_t i = 0; !err && i < inputs.size(); i += 2) {
        err = !static_cast<AbstractDict *>(res)->MapAdd(inputs[i], inputs[i + 1]);
      }
      break;
    case BUILD_STRING:
      res = MakeAObject(kTypeString);
      break;
    case BUILD_SLICE:
      res = MakeAObject(kTypeSlice);
      break;
    case BUILD_SET:
      res = MakeAObject(kTypeSet);
      break;
    default:
      err = true;
      break;
  }
  return err ? MakeAObject(kTypeAnyValue) : res;
}

AObject *AbstractObjectBase::MergeOperations(AObject *container, std::vector<AObject *> args, int opcode) {
  Type type = container ? container->GetType() : kTypeAnyValue;
  bool success = false;
  switch (opcode) {
    case LIST_EXTEND:
      success = type == kTypeList && (static_cast<AbstractList *>(container))->ListExtend(args[0]);
      break;
    case LIST_APPEND:
      success = type == kTypeList && (static_cast<AbstractList *>(container))->ListAppend(args[0]);
      break;
    case DICT_MERGE:
      success = type == kTypeDict && (static_cast<AbstractDict *>(container))->DictMerge(args[0]);
      break;
    case DICT_UPDATE:
      success = type == kTypeDict && (static_cast<AbstractDict *>(container))->DictUpdate(args[0]);
      break;
    case MAP_ADD:
      success = type == kTypeDict && (static_cast<AbstractDict *>(container))->MapAdd(args[0], args[1]);
      break;
    case SET_UPDATE: /* fall-through */
    case SET_ADD:
      success = true;
      container = MakeAObject(kTypeSet);
      break;
    default:
      break;
  }
  if (!success) {
    return MakeAObject(kTypeAnyValue);
  }
  return container;
}

AbstractObject::AbstractObject(Type type, const py::object &o) : AbstractObjectBase(type), value_(o) {
  // cache attr
  (void)GetAttr("__ms_mutable__");
}

AObject *AbstractObject::GetIter() const {
  if (this->GetType() == kTypeAnyValue || value_.ptr() == nullptr) {
    return MakeAObject(kTypeAnyValue);
  }
  PyObject *iter = PyObject_GetIter(value_.ptr());
  CHECK_PYTHON_EXCEPTION(iter);
  AObject *res = Convert(iter);
  Py_XDECREF(iter);
  return res;
}

AObject *AbstractObjectBase::GetAttr(const std::string &name) {
  PyTypeObject *tp = type_object_;
  if (tp == nullptr) {
    return MakeAObject(kTypeAnyValue);
  }
  py::str name_obj(name);
  PyObject *attr_obj = PyObject_GetAttr(reinterpret_cast<PyObject *>(tp), name_obj.ptr());
  if (attr_obj == nullptr) {
    PyErr_Clear();
    return MakeAObject(kTypeAnyValue);
  }
  AObject *attr = AObject::Convert(attr_obj);
  Py_DECREF(attr_obj);

  // look up mro, borrowed
  PyObject *descr = _PyType_Lookup(tp, name_obj.ptr());
  if (descr) {
    // check @staticmethod and @classmethod
    if (Py_IS_TYPE(descr, &PyStaticMethod_Type) || Py_IS_TYPE(descr, &PyClassMethod_Type)) {
      // attr not modify
    } else if (PyFunction_Check(descr)) {
      MS_EXCEPTION_IF_CHECK_FAIL(attr_obj == descr, "unknown user defined descriptor");
      PyObject *meth = PyMethod_New(descr, Py_None);
      AObject *m = AObject::Convert(meth);
      Py_DECREF(meth);
      m->SetAttr("__self__", this);
      m->SetAttr("__func__", attr);
      attr = m;
    } else {
      // other type
      attr = nullptr;
    }
  }
  return attr;
}

AObject *AbstractObject::GetAttr(const std::string &name) {
  FIND_MAP_CACHE(attrs_, name);
  AObject *res = nullptr;
  if (value_.ptr() != nullptr) {
    PRINT_IF_HAS_USER_DEFINED_HOOK(value_.ptr(), __getattr__);
    PRINT_IF_HAS_USER_DEFINED_HOOK(value_.ptr(), __getattribute__);
#ifdef DEBUG
    PyObject *tmp = PyObject_GetAttrString(reinterpret_cast<PyObject *>(Py_TYPE(value_.ptr())), name.c_str());
    if (tmp) {  // is user defined descriptor ?
      PRINT_IF_HAS_USER_DEFINED_HOOK(tmp, __get__);
    } else {
      PyErr_Clear();
    }
    Py_XDECREF(tmp);
#endif
    PyObject *attr = PyObject_GetAttrString(value_.ptr(), name.c_str());
    CHECK_PYTHON_EXCEPTION(attr);
    res = Convert(attr);
    Py_XDECREF(attr);
  } else {
    res = this->AbstractObjectBase::GetAttr(name);
  }
  attrs_[name] = res;
  return res;
}

bool AbstractObject::SetAttr(const std::string &n, AObject *v) {
  attrs_[n] = v ? v : MakeAObject(kTypeAnyValue);
  return true;
}

AObject *AbstractSequence::GetItem(AObject *k) {
  auto iter = write_cache_.find(k);
  if (iter != write_cache_.end()) {
    return iter->second == nullptr ? MakeAObject(kTypeAnyValue) : iter->second;
  }
  return this->AbstractObject::GetItem(k);
}

AObject *AbstractObject::GetItem(AObject *k) {
  PyObject *s = this->GetPyObject().ptr();
  PyObject *i = k ? k->GetPyObject().ptr() : nullptr;
  PyObject *t = nullptr;
  if (s != nullptr && i != nullptr && k->GetType() != kTypeAnyValue) {
    t = PyObject_GetItem(s, i);
    CHECK_PYTHON_EXCEPTION(t);
  }
  AObject *res = Convert(t);
  Py_XDECREF(t);
  return res;
}

bool AbstractSequence::SetItem(AObject *k, AObject *v) {
  if (this->type_ == kTypeString || this->type_ == kTypeTuple) {
    return false;
  }
  write_cache_[k] = v ? v : MakeAObject(kTypeAnyValue);
  return true;
}

AObject *AbstractObject::UnaryValue(int op) const {
  PyObject *res = nullptr;
  switch (op) {
    case UNARY_POSITIVE:
      res = PyNumber_Positive(value_.ptr());
      break;
    case UNARY_NEGATIVE:
      res = PyNumber_Negative(value_.ptr());
      break;
    case UNARY_INVERT:
      res = PyNumber_Invert(value_.ptr());
      break;
    case UNARY_NOT: {
      int err = PyObject_IsTrue(value_.ptr());
      res = err > 0 ? Py_False : (err == 0 ? Py_True : nullptr);
      break;
    }
    default:
      break;
  }
  CHECK_PYTHON_EXCEPTION(res);
  AObject *ret = Convert(res);
  Py_XDECREF(res);
  return ret;
}

AObject *AbstractObject::Unary(int op) const {
  if (this->GetType() == kTypeAnyValue) {
    return MakeAObject(kTypeAnyValue);
  }
  if (value_.ptr() != nullptr) {
    return UnaryValue(op);
  }
  Type res_type = kTypeAnyValue;
  Type type = this->GetType();
  switch (op) {
    case UNARY_POSITIVE:
    case UNARY_NEGATIVE:
    case UNARY_INVERT:
      if (type == kTypeBool || type == kTypeInt) {
        res_type = kTypeInt;
      } else if (type == kTypeFloat) {
        res_type = kTypeFloat;
      }
      break;
    case UNARY_NOT: {
      bool is_num = type == kTypeBool || type == kTypeInt || type == kTypeFloat;
      if (is_num || type == kTypeList || type == kTypeTuple || type == kTypeDict) {
        res_type = kTypeBool;
      }
      break;
    }
    default:
      break;
  }
  return MakeAObject(res_type);
}

static PyObject *BinaryPow(PyObject *base, PyObject *exp) { return PyNumber_Power(base, exp, Py_None); }
static PyObject *InplacePow(PyObject *base, PyObject *exp) { return PyNumber_InPlacePower(base, exp, Py_None); }

static AObject::Type BinaryIntOp(AObject::Type l, AObject::Type r) {
  AObject::Type type = AObject::kTypeAnyValue;
  switch (l) {
    case AObject::kTypeInt:
    case AObject::kTypeBool:
      if (r == AObject::kTypeInt || r == AObject::kTypeBool) {
        type = AObject::kTypeInt;
      }
      break;
    default:
      break;
  }
  return type;
}

// operator '&', '^', '|'
static AObject::Type NumberLogic(AObject::Type l, AObject::Type r) {
  AObject::Type type = AObject::kTypeAnyValue;
  if (l == AObject::kTypeBool) {
    if (r == AObject::kTypeInt || r == AObject::kTypeBool) {
      type = r;
    }
  } else {
    type = BinaryIntOp(l, r);
  }
  return type;
}

// operator '+', '-', '*', '/', '%', '**', '//'
static AObject::Type NumberArithmetic(AObject::Type l, AObject::Type r) {
  AObject::Type type = AObject::kTypeAnyValue;
  if (l == AObject::kTypeFloat || r == AObject::kTypeFloat) {
    if (l == AObject::kTypeInt || l == AObject::kTypeBool || r == AObject::kTypeInt || r == AObject::kTypeBool) {
      type = AObject::kTypeFloat;
    }
  } else {
    type = BinaryIntOp(l, r);
  }
  return type;
}

static AObject::Type BinaryAdd(AObject::Type l, AObject::Type r) {
  AObject::Type type = AObject::kTypeAnyValue;
  switch (l) {
    case AObject::kTypeTuple:
    case AObject::kTypeList:
    case AObject::kTypeString:
      if (r == l) {
        type = l;
      }
      break;
    default:
      type = NumberArithmetic(l, r);
      break;
  }
  return type;
}

static AObject::Type BinaryInferDefault(AObject::Type, AObject::Type) { return AObject::kTypeAnyValue; }

static AObject *BinaryIs(AObject *l, AObject *r) {
  PyObject *a = l ? l->GetPyObject().ptr() : nullptr;
  PyObject *b = r ? r->GetPyObject().ptr() : nullptr;
  const auto &map = const_object_type_map;
  bool const_a = map.find(a) != map.end();
  bool const_b = map.find(b) != map.end();
  // all is const object
  if (const_a && const_b) {
    return AObject::Convert(a == b ? Py_True : Py_False);
  }
  // a const object and a known object
  if ((const_a && b) || (const_b && a)) {
    return AObject::Convert(Py_False);
  }
  // a const object and a unknown object, but known it's type
  if (const_a && r != nullptr && r->GetType() != AObject::kTypeAnyValue && r->GetType() != AObject::kTypeBool) {
    MS_EXCEPTION_IF_CHECK_FAIL(!const_b, "shouldn't reach here");
    return AObject::Convert(Py_False);
  }
  if (const_b && l != nullptr && l->GetType() != AObject::kTypeAnyValue && l->GetType() != AObject::kTypeBool) {
    MS_EXCEPTION_IF_CHECK_FAIL(!const_a, "shouldn't reach here");
    return AObject::Convert(Py_False);
  }
  return AObject::MakeAObject(AObject::kTypeBool);
}

static AObject *BinaryContains(AObject *l, AObject *r) {
  PyObject *o = l->GetPyObject().ptr();
  PyObject *c = r->GetPyObject().ptr();
  if (c == nullptr || o == nullptr || r->GetType() == AObject::kTypeAnyValue) {
    return AObject::MakeAObject(AObject::kTypeBool);
  }
  int res = PySequence_Contains(c, o);
  CHECK_PYTHON_EXCEPTION(res < 0 ? nullptr : Py_True);
  return AObject::Convert(res ? Py_True : Py_False);
}

using InferBinaryFunc = AObject *(*)(AObject *, AObject *);
using InferBinaryTypeFunc = AObject::Type (*)(AObject::Type, AObject::Type);

template <binaryfunc pyfunc, InferBinaryTypeFunc type_infer>
AObject *InferBinary(AObject *a, AObject *b) {
  PyObject *l = a->GetPyObject().ptr();
  PyObject *r = b->GetPyObject().ptr();
  if (l == nullptr || r == nullptr) {
    return AObject::MakeAObject(type_infer(a->GetType(), b->GetType()));
  }
  if (a->GetType() == AObject::kTypeAnyValue || b->GetType() == AObject::kTypeAnyValue) {
    return AObject::MakeAObject(AObject::kTypeAnyValue);
  }
  PyObject *o = pyfunc(l, r);
  CHECK_PYTHON_EXCEPTION(o);
  AObject *res = AObject::Convert(o);
  Py_XDECREF(o);
  return res;
}

// the inplace binary operations of known type don't modify original python object
// list, tuple, dict already override binary
static std::unordered_map<int, InferBinaryFunc> infer_binary_func = {
  {BINARY_MATRIX_MULTIPLY, InferBinary<PyNumber_MatrixMultiply, BinaryInferDefault>},          // '@'
  {INPLACE_MATRIX_MULTIPLY, InferBinary<PyNumber_InPlaceMatrixMultiply, BinaryInferDefault>},  // '@='
  {BINARY_POWER, InferBinary<BinaryPow, NumberArithmetic>},                                    // '**'
  {INPLACE_POWER, InferBinary<InplacePow, NumberArithmetic>},                                  // '**='
  {BINARY_MULTIPLY, InferBinary<PyNumber_Multiply, NumberArithmetic>},                         // '*'
  {INPLACE_MULTIPLY, InferBinary<PyNumber_InPlaceMultiply, NumberArithmetic>},                 // '*='
  {BINARY_MODULO, InferBinary<PyNumber_Remainder, NumberArithmetic>},                          // '%'
  {INPLACE_MODULO, InferBinary<PyNumber_InPlaceRemainder, NumberArithmetic>},                  // '%='
  {BINARY_ADD, InferBinary<PyNumber_Add, BinaryAdd>},
  {INPLACE_ADD, InferBinary<PyNumber_InPlaceAdd, BinaryAdd>},
  {BINARY_SUBTRACT, InferBinary<PyNumber_Subtract, NumberArithmetic>},
  {INPLACE_SUBTRACT, InferBinary<PyNumber_InPlaceSubtract, NumberArithmetic>},
  {BINARY_FLOOR_DIVIDE, InferBinary<PyNumber_FloorDivide, NumberArithmetic>},          // '//'
  {INPLACE_FLOOR_DIVIDE, InferBinary<PyNumber_InPlaceFloorDivide, NumberArithmetic>},  // '//='
  {BINARY_TRUE_DIVIDE, InferBinary<PyNumber_TrueDivide, NumberArithmetic>},
  {INPLACE_TRUE_DIVIDE, InferBinary<PyNumber_InPlaceTrueDivide, NumberArithmetic>},
  {BINARY_LSHIFT, InferBinary<PyNumber_Lshift, BinaryIntOp>},
  {INPLACE_LSHIFT, InferBinary<PyNumber_InPlaceLshift, BinaryIntOp>},
  {BINARY_RSHIFT, InferBinary<PyNumber_Rshift, BinaryIntOp>},
  {INPLACE_RSHIFT, InferBinary<PyNumber_InPlaceRshift, BinaryIntOp>},
  {BINARY_AND, InferBinary<PyNumber_And, NumberLogic>},
  {INPLACE_AND, InferBinary<PyNumber_InPlaceAnd, NumberLogic>},
  {BINARY_XOR, InferBinary<PyNumber_Xor, NumberLogic>},
  {INPLACE_XOR, InferBinary<PyNumber_InPlaceXor, NumberLogic>},
  {BINARY_OR, InferBinary<PyNumber_Or, NumberLogic>},
  {INPLACE_OR, InferBinary<PyNumber_InPlaceOr, NumberLogic>},
  {CONTAINS_OP, BinaryContains},
  {IS_OP, BinaryIs}};

AObject *AbstractObject::Binary(AObject *other, int op) {
  if (other == nullptr) {
    return MakeAObject(kTypeAnyValue);
  }
  auto iter = infer_binary_func.find(op);
  return iter == infer_binary_func.end() ? MakeAObject(kTypeAnyValue) : iter->second(this, other);
}

AObject *AbstractType::BuildAbstractInstance(const std::vector<AObject *> &args, int opcode) {
  Type type = kTypeAnyValue;
  PyTypeObject *tp = reinterpret_cast<PyTypeObject *>(value_.ptr());
  AbstractTuple *res;
  switch (type_type_) {
    case kTypeList:
    case kTypeTuple:
      res = static_cast<AbstractTuple *>(MakeAObject(type_type_));
      if (tp != &PyTuple_Type) {
        return res;
      }
      if (args.size() == 0) {
        res->Update(args);
        return res;
      }
      if (args[0] && (args[0]->GetType() == kTypeTuple || args[0]->GetType() == kTypeList)) {
        res->Update(static_cast<AbstractTuple *>(args[0])->items());
        return res;
      }
      return res;
    case kTypeBool:
      if (args.size() == 0) {
        return Convert(Py_False);
      }
      type = args[0] ? args[0]->GetType() : kTypeAnyValue;
      if (type == kTypeList || type == kTypeTuple) {
        AbstractTuple *tmp = static_cast<AbstractTuple *>(args[0]);
        return tmp->IsElementValid() ? Convert(tmp->size() ? Py_True : Py_False) : MakeAObject(kTypeBool);
      }
      if (type == kTypeDict) {
        AbstractDict *tmp = static_cast<AbstractDict *>(args[0]);
        return tmp->IsElementValid() ? Convert(tmp->size() ? Py_True : Py_False) : MakeAObject(kTypeBool);
      }
      break;
    default:
      break;
  }
  return MakeAObject(type_type_, tp, nullptr);
}

// this function call object without error
py::object AbstractType::BuildInstance(const std::vector<py::object> &args, int opcode) {
  if (value_.ptr() == nullptr) {
    MS_LOG(DEBUG) << "create instance failed, unknown class";
    return py::object();
  }
  auto pair = Utils::PackCallStackArgs(args, opcode, true);
  if (pair.first.ptr() == nullptr) {
    MS_LOG(DEBUG) << "create instance failed, unknown opcode or arguments";
    return py::object();
  }
  PyObject *const *vector_args = &PyTuple_GET_ITEM(pair.first.ptr(), 0);
  Py_ssize_t kw_cnt = pair.second.ptr() == nullptr ? 0 : PyTuple_GET_SIZE(pair.second.ptr());
  Py_ssize_t nargs = PyTuple_GET_SIZE(pair.first.ptr());
  PyObject *inst = PyObject_Vectorcall(value_.ptr(), vector_args, nargs - kw_cnt, pair.second.ptr());
  CHECK_PYTHON_EXCEPTION(inst);
  return py::reinterpret_steal<py::object>(inst);
}

AObject *AbstractTuple::Binary(AObject *o, int op) {
  // generic binary
  PyObject *r_obj = o ? o->GetPyObject().ptr() : nullptr;
  if (op == IS_OP) {
    bool cnst = const_object_type_map.find(r_obj) != const_object_type_map.end();
    return cnst ? Convert(Py_False) : MakeAObject(kTypeBool);
  }
  if (op == CONTAINS_OP) {
    return infer_binary_func[CONTAINS_OP](this, o);
  }
  // tuple binary
  if (o == nullptr || this->GetType() != o->GetType()) {
    if (this->GetType() == kTypeList && op == BINARY_MULTIPLY && (o != nullptr) && o->GetType() == kTypeInt) {
      AbstractTuple *ret = static_cast<AbstractTuple *>(MakeAObject(this->GetType()));
      std::vector<AObject *> temp;
      int res = PyLong_AsLong(o->GetPyObject().ptr());
      for (int i = 0; i < res; i++) {
        std::copy(items_.begin(), items_.end(), std::back_inserter(temp));
      }
      ret->Update(std::move(temp));
      return ret;
    }
    return MakeAObject(kTypeAnyValue);
  }
  AbstractTuple *r_list = static_cast<AbstractTuple *>(o);
  if (!this->IsElementValid() || !r_list->IsElementValid()) {
    return MakeAObject(kTypeAnyValue);
  }
  if (op == BINARY_ADD || (this->GetType() == kTypeTuple && op == INPLACE_ADD)) {
    AbstractTuple *ret = static_cast<AbstractTuple *>(MakeAObject(this->GetType()));
    std::vector<AObject *> temp;
    std::copy(items_.begin(), items_.end(), std::back_inserter(temp));
    std::copy(r_list->items_.begin(), r_list->items_.end(), std::back_inserter(temp));
    ret->Update(std::move(temp));
    return ret;
  }
  if (op == INPLACE_ADD) {
    std::copy(r_list->items_.begin(), r_list->items_.end(), std::back_inserter(this->items_));
    MarkModify();
    return this;
  }
  // binary mul, inplace mul
  return MakeAObject(kTypeAnyValue);
}

AObject *AbstractTuple::Unary(int op) const {
  if (op != UNARY_NOT || !this->IsElementValid()) {
    return MakeAObject(kTypeAnyValue);
  }
  return Convert(this->size() > 0 ? Py_True : Py_False);
}

#define RECURSION_CONVERT(iter_expr, get_expr, set_expr, item)   \
  RecMap holder;                                                 \
  if (rec == nullptr) {                                          \
    rec = &holder;                                               \
  }                                                              \
  (*rec)[seq.ptr()] = this;                                      \
  AObject *aobject = nullptr;                                    \
  PyObject *item = nullptr;                                      \
  iter_expr {                                                    \
    get_expr;                                                    \
    auto iter = rec->find(item);                                 \
    if (iter != rec->end()) {                                    \
      aobject = iter->second;                                    \
    } else {                                                     \
      Type t = GetPyType(item);                                  \
      if (t == kTypeList || t == kTypeTuple || t == kTypeDict) { \
        PyTypeObject *tp = Py_TYPE(item);                        \
        aobject = MakeAObject(t, tp, item, rec);                 \
      } else {                                                   \
        aobject = Convert(item);                                 \
      }                                                          \
    }                                                            \
    set_expr;                                                    \
  }

AbstractTuple::AbstractTuple(Type type, py::object seq, RecMap *rec)
    : AbstractSequence(type, seq),
      items_(),
      ms_support_(kBoolUnknown),
      element_type_(kTypeAnyValue),
      element_valid_(false),
      modify_(false) {
  type_object_ = (type == kTypeList) ? &PyList_Type : &PyTuple_Type;
  if (!seq.ptr()) {
    return;
  }
  element_valid_ = true;
  MS_EXCEPTION_IF_CHECK_FAIL(GetPyType(seq.ptr()) == type, std::string("convert ") + GetTypeDesc(type) + " but got " +
                                                             GetTypeDesc(GetPyType(seq.ptr())));
  PyObject *o = seq.ptr();
  Py_ssize_t siz = Py_SIZE(seq.ptr());
  items_.resize(siz);

#define ITER_EXPR for (int i = 0; i < siz; ++i)
#define GET_EXPR (item = (type == kTypeList) ? PyList_GET_ITEM(o, i) : PyTuple_GET_ITEM(o, i))
#define SET_EXPR items_[i] = aobject
  RECURSION_CONVERT(ITER_EXPR, GET_EXPR, SET_EXPR, item);
#undef ITER_EXPR
#undef GET_EXPR
#undef SET_EXPR

  // copy it
  Update();
}

AbstractDict::AbstractDict(Type type, py::object seq, RecMap *rec)
    : AbstractSequence(type, seq),
      dict_(),
      k_type_(kTypeAnyValue),
      v_type_(kTypeAnyValue),
      element_valid_(false),
      modify_(false) {
  type_object_ = &PyDict_Type;
  if (!seq.ptr()) {
    return;
  }
  element_valid_ = true;
  MS_EXCEPTION_IF_CHECK_FAIL(GetPyType(seq.ptr()) == type, std::string("convert ") + GetTypeDesc(type) + ", but got " +
                                                             GetTypeDesc(GetPyType(seq.ptr())));
  PyObject *m = dict_.ptr();
  PyObject *k;
  Py_ssize_t p = 0;

#define ITER_EXPR while (PyDict_Next(seq.ptr(), &p, &k, &item))
#define GET_EXPR PRINT_IF_HAS_USER_DEFINED_HOOK(k, __hash__)
#define SET_EXPR PyDict_SetItem(m, k, ConvertValue(aobject).ptr())
  RECURSION_CONVERT(ITER_EXPR, GET_EXPR, SET_EXPR, item);
#undef ITER_EXPR
#undef SET_EXPR
#undef GET_EXPR

  // copy it
  Update();
}

#undef RECURSION_CONVERT

bool AbstractTuple::IsMindSporeSupportedType() {
  if (ms_support_ != kBoolUnknown) {
    return ms_support_ == kBoolTrue;
  }
  ms_support_ = kBoolFalse;
  if (kMsSupportedType.find(element_type_) != kMsSupportedType.end()) {
    ms_support_ = kBoolTrue;
    return true;
  }
  if (!this->IsElementValid()) {
    return false;
  }
  for (auto i : *this) {
    if (!i) {
      return false;
    }
    if (!i->IsMindSporeSupportedType()) {
      return false;
    }
  }
  ms_support_ = kBoolTrue;
  return true;
}

bool AbstractDict::IsMindSporeSupportedType() {
  if (kMsSupportedType.find(k_type_) != kMsSupportedType.end() &&
      kMsSupportedType.find(v_type_) != kMsSupportedType.end()) {
    return true;
  }
  if (this->IsElementValid()) {
    for (auto i : *this) {
      if (!i) {
        return false;
      }
      Type t = i->GetType();
      if (t == kTypeList || t == kTypeTuple || t == kTypeDict) {
        // check self reference object
        return false;
      }
      if (!i->IsMindSporeSupportedType()) {
        return false;
      }
    }
    return true;
  }
  return false;
}

std::string AbstractTuple::ToString() const {
  std::stringstream s;
  s << this->AObject::ToString() << "<" << GetTypeDesc(element_type_) << ">";
  if (this->IsElementValid()) {
    s << " size:" << this->size();
  } else {
    s << "<NoSize>";
  }
  return s.str();
}

std::string AbstractDict::ToString() const {
  std::stringstream s;
  s << this->AObject::ToString() << '<' << GetTypeDesc(k_type_) << ',' << GetTypeDesc(v_type_) << '>';
  if (this->IsElementValid()) {
    s << " size:" << size();
  } else {
    s << "<NoSize>";
  }
  return s.str();
}

/**
 * cast to Py_ssize_t, call hook __index__ by PyNumber_AsSsize_t
 * \return -1 if key error, out of bound, overflow to cast Py_ssize_t
 */
static Py_ssize_t GetTupleIndex(AObject *k, Py_ssize_t size) {
  Py_ssize_t index = PyLong_AsSsize_t(k->GetPyObject().ptr());
  if (PyErr_Occurred()) {
    PyErr_Clear();
    return -1;
  }
  if (index < -size || index >= size) {
    return -1;
  }
  index = index < 0 ? (size + index) : index;
  return index;
}

bool AbstractList::SetItem(AObject *k, AObject *v) {
  MarkModify();
  if (k == nullptr || k->GetType() == AObject::kTypeAnyValue || k->GetPyObject().ptr() == nullptr) {
    // user defined index or unknown key
    this->AbstractSequence::SetItem(k, v);
    return true;
  }
  if (!IsElementValid()) {
    return true;
  }
  Py_ssize_t index = GetTupleIndex(k, this->size());
  if (index == -1) {
    MarkElementInValid();
    return false;
  }
  items_[index] = v;
  element_type_ = v->GetType() == element_type_ ? element_type_ : kTypeAnyValue;
  return true;
}

AObject *AbstractTuple::GetItem(AObject *k) {
  if (k == nullptr || k->GetType() == AObject::kTypeAnyValue || k->GetPyObject().ptr() == nullptr) {
    // user defined index or unknown key
    return this->AbstractSequence::GetItem(k);
  }
  if (!IsElementValid()) {
    return AObject::MakeAObject(element_type_);
  }
  if (k->GetType() == AObject::kTypeSlice) {
    if (this->GetPyObject().ptr() != nullptr) {
      return this->AbstractSequence::GetItem(k);
    }
    AObject *resultTuple = AObject::MakeAObject(this->type_);
    PyObject *slicePyObject = k->GetPyObject().ptr();
    Py_ssize_t start, stop, step;
    if (PySlice_Unpack(slicePyObject, &start, &stop, &step) < 0) {
      return AObject::MakeAObject(kTypeAnyValue);
    }
    if (start >= stop) {
      return resultTuple;
    }
    Py_ssize_t sliceLength = PySlice_AdjustIndices(this->items().size(), &start, &stop, step);
    AbstractTuple *resultTuplePtr = static_cast<AbstractTuple *>(resultTuple);
    if (start == 0 && step == 1 && sliceLength == this->size()) {
      return this;
    }
    if (step > 1) {
      int cursor = 0;
      std::vector<AObject *> itemsVector;
      for (cursor = 0; cursor < stop; cursor += step) {
        itemsVector.push_back(this->items()[cursor]);
      }
      resultTuplePtr->Update(itemsVector);
      return resultTuplePtr;
    }
    return AObject::MakeAObject(kTypeAnyValue);
  }
  Py_ssize_t index = GetTupleIndex(k, this->size());
  if (index == -1) {
    return AObject::MakeAObject(kTypeAnyValue);
  }
  return items_[index];
}

#undef GET_INDEX

AObject *AbstractTuple::GetAttr(const std::string &name) {
  py::object list = (type_ == kTypeList) ? (py::object)py::list() : py::tuple();
  PyObject *attr = PyObject_GetAttrString(list.ptr(), name.c_str());
  CHECK_PYTHON_EXCEPTION(attr);
  if (attr == nullptr) {
    FIND_MAP_CACHE(attrs_, name);
  }
  AObject *res = Convert(attr);
  Py_XDECREF(attr);
  return res;
}

AObject *AbstractDict::Unary(int op) const {
  if (op != UNARY_NOT || !this->IsElementValid()) {
    return MakeAObject(kTypeAnyValue);
  }
  return Convert(this->size() ? Py_True : Py_False);
}

AObject *AbstractDict::Binary(AObject *other, int op) {
  if (op == IS_OP) {
    PyObject *b = other ? other->GetPyObject().ptr() : nullptr;
    bool cnst = const_object_type_map.find(b) != const_object_type_map.end();
    return cnst ? Convert(Py_False) : MakeAObject(kTypeBool);
  }
  if (op == CONTAINS_OP && other != nullptr) {
    return infer_binary_func[CONTAINS_OP](this, other);
  }
  return MakeAObject(kTypeAnyValue);
}

AObject *AbstractDict::GetAttr(const std::string &name) {
  if (value_.ptr() == nullptr) {
    return AObject::MakeAObject(kTypeAnyValue);
  }
  PyObject *attr = PyObject_GetAttrString(value_.ptr(), name.c_str());
  CHECK_PYTHON_EXCEPTION(attr);
  AObject *res = Convert(attr);
  Py_XDECREF(attr);
  return res;
}

AObject *AbstractDict::GetItem(AObject *k) {
  auto iter = this->write_cache_.find(k);
  if (iter != this->write_cache_.end()) {
    return iter->second == nullptr ? MakeAObject(kTypeAnyValue) : iter->second;
  }
  if (!IsElementValid()) {
    return MakeAObject(v_type_);
  }
  PyObject *key = k ? k->GetPyObject().ptr() : nullptr;
  if (key == nullptr) {
    return MakeAObject(kTypeAnyValue);
  }
  PRINT_IF_HAS_USER_DEFINED_HOOK(key, __hash__);
  PyObject *item = PyDict_GetItem(dict_.ptr(), key);
  return item == nullptr ? MakeAObject(kTypeAnyValue) : ConvertValue(item);
}

bool AbstractDict::DictMerge(AObject *o, int update) {
  MarkModify();
  if (!IsElementValid()) {
    return true;
  }
  if (o == nullptr || o->GetType() != kTypeDict) {
    MarkElementInValid();
    return true;
  }
  AbstractDict *d = static_cast<AbstractDict *>(o);
  if (!d->IsElementValid() || PyDict_Merge(dict_.ptr(), d->dict_.ptr(), update)) {
    MarkElementInValid();
    CHECK_PYTHON_EXCEPTION(nullptr);
    // unknown user defined dict merge, assume it success
  }
  if (size() == 0) {
    this->k_type_ = d->k_type_;
    this->v_type_ = d->v_type_;
  } else {
    this->k_type_ = this->k_type_ == d->k_type_ ? this->k_type_ : kTypeAnyValue;
    this->v_type_ = this->v_type_ == d->v_type_ ? this->v_type_ : kTypeAnyValue;
  }
  return true;
}

bool AbstractDict::DictUpdate(AObject *o) { return DictMerge(o, 1); }

bool AbstractDict::MapAdd(AObject *k, AObject *v) {
  if (v == nullptr) {
    MarkElementInValid();
    return true;  // assume it success
  }
  if (size() == 0) {
    this->k_type_ = k->GetType();
    this->v_type_ = v->GetType();
  } else {
    this->k_type_ = this->k_type_ == k->GetType() ? this->k_type_ : kTypeAnyValue;
    this->v_type_ = this->v_type_ == v->GetType() ? this->v_type_ : kTypeAnyValue;
  }
  return SetItem(k, v);
}

bool AbstractList::ListAppend(AObject *item) {
  MarkModify();
  if (!IsElementValid()) {
    return true;
  }
  if (size() == 0) {
    this->element_type_ = item->GetType();
  } else if (this->element_type_ != item->GetType()) {
    this->element_type_ = kTypeAnyValue;
  }
  items_.push_back(item);
  return true;
}

bool AbstractList::ListExtend(AObject *l) {
  MarkModify();
  if (!IsElementValid()) {
    return true;
  }
  if (l == nullptr || (l->GetType() != kTypeTuple && l->GetType() != kTypeList)) {
    MarkElementInValid();
    return true;
  }
  AbstractTuple *i = static_cast<AbstractTuple *>(l);
  if (!i->IsElementValid()) {
    MarkElementInValid();
    return true;
  }
  if (size() == 0) {
    this->element_type_ = i->GetElementType();
  } else {
    this->element_type_ = this->GetElementType() == i->GetElementType() ? this->GetElementType() : kTypeAnyValue;
  }
  std::copy(i->items().begin(), i->items().end(), std::back_inserter(items_));
  return true;
}

AbstractTuple *AbstractList::ListToTuple() {
  AbstractTuple *res = static_cast<AbstractTuple *>(MakeAObject(kTypeTuple));
  if (!IsElementValid()) {
    return res;
  }
  res->SetElementType(this->element_type_);
  res->Update(this->items_);
  return res;
}

bool AbstractTuple::Update(const std::vector<AObject *> &item) {
  this->element_valid_ = true;
  this->items_ = item;
  if (this->items_.size() != 0 && items_[0] != nullptr) {
    this->element_type_ = items_[0]->GetType();
    bool any = item.end() != std::find_if(item.begin(), item.end(), [this](AObject *i) {
                 return i ? i->GetType() != this->element_type_ : true;
               });
    this->element_type_ = any ? kTypeAnyValue : this->element_type_;
  }
  return Update();
}

bool AbstractTuple::Update() {
  if (!this->IsElementValid()) {
    return false;
  }
  this->element_type_ = kTypeAnyValue;
  // copy it
  PyObject *c = (this->type_ == kTypeTuple) ? PyTuple_New(items_.size()) : PyList_New(items_.size());
  modify_ = false;
  value_ = py::reinterpret_steal<py::object>(c);
  for (size_t i = 0; i < items_.size(); i++) {
    py::object item = (items_[i] != nullptr) ? items_[i]->GetPyObject() : py::object();
    if (item.ptr() == nullptr) {
      value_ = py::object();
      return false;
    }
    if (this->type_ == kTypeTuple) {
      PyTuple_SET_ITEM(c, i, item.inc_ref().ptr());
    } else {
      PyList_SET_ITEM(c, i, item.inc_ref().ptr());
    }
    if (i == 0) {
      this->element_type_ = items_[i]->GetType();
    } else {
      this->element_type_ = this->element_type_ == items_[i]->GetType() ? this->element_type_ : kTypeAnyValue;
    }
  }
  return true;
}

py::object AbstractList::GetPyObject() {
  if (this->write_cache_.size()) {
    // see SetItem, can't update unknown value to list
    return py::object();
  }
  if (modify_ && !Update()) {
    return py::object();
  }
  return value_;
}

bool AbstractDict::Update() {
  value_ = py::object();
  for (auto i : this->write_cache_) {
    PyObject *key = i.first == nullptr ? nullptr : i.first->GetPyObject().ptr();
    if (key == nullptr || -1 == PyDict_SetItem(dict_.ptr(), key, ConvertValue(i.second).ptr())) {
      MarkElementInValid();
      PyErr_Clear();
      return false;
    }
  }
  this->write_cache_.clear();
  // copy it
  value_ = py::dict();
  PyObject *k, *v;
  Py_ssize_t p = 0;
  bool init_element_type = false;
  while (PyDict_Next(dict_.ptr(), &p, &k, &v)) {
    AObject *i = ConvertValue(v);
    py::object item = i != nullptr ? i->GetPyObject() : py::object();
    if (item.ptr() == nullptr) {
      value_ = py::object();
      break;
    }
    PyDict_SetItem(value_.ptr(), k, item.ptr());
    if (init_element_type) {
      k_type_ = k_type_ == GetPyType(k) ? k_type_ : kTypeAnyValue;
      v_type_ = v_type_ == i->GetType() ? v_type_ : kTypeAnyValue;
    } else {
      k_type_ = GetPyType(k);
      v_type_ = i->GetType();
      init_element_type = true;
    }
  }
  return true;
}

py::object AbstractDict::GetPyObject() {
  if (!IsElementValid()) {
    return py::object();
  }
  if (!IsModify()) {
    return value_;
  }
  Update();
  return value_;
}

py::object AbstractTensor::GetTensor(bool sync) {
  if (!is_stub_ || !sync) {
    return value_;
  }
  std::string attr_key = "tensor";
  auto iter = attrs_.find(attr_key);
  if (iter != attrs_.end()) {
    return iter->second->GetPyObject();
  }
  PyObject *res = PyObject_GetAttrString(value_.ptr(), attr_key.c_str());
  if (res != nullptr && res != Py_None) {
    attrs_[attr_key] = AObject::Convert(res);
    return py::reinterpret_steal<py::object>(res);
  }
  if (res == nullptr) {
    PyErr_Clear();
  } else {
    Py_DECREF(res);
  }
  PyObject *meth = PyObject_GetAttrString(value_.ptr(), "stub_sync");
  MS_EXCEPTION_IF_CHECK_FAIL(meth && PyMethod_Check(meth), "check value");
  res = PyObject_Call(meth, py::tuple().ptr(), nullptr);
  Py_DECREF(meth);
  CHECK_PYTHON_EXCEPTION(res);
  attrs_[attr_key] = AObject::Convert(res);
  return py::reinterpret_steal<py::object>(res);
}

static bool CheckAdapterTensor(py::object tensor) {
  bool is_adapter = false;
  if (IsStubTensor(tensor)) {
    is_adapter = py::hasattr(tensor, "adapter_flag") && py::cast<bool>(py::getattr(tensor, "adapter_flag"));
  } else {
    is_adapter = py::cast<mindspore::tensor::TensorPtr>(tensor.ptr())->is_adapter();
  }
  return is_adapter;
}

AbstractBasePtr PyObjectToAbstract(const py::object &arg) {
  ValuePtr converted = nullptr;
  bool success;
  if (IsStubTensor(arg)) {
    success = mindspore::parse::ConvertStubData(arg, &converted);
  } else {
    success = mindspore::parse::ConvertData(arg, &converted);
  }
  if (!success) {
    MS_LOG(EXCEPTION) << "Fail to convert the object: " << py::str(arg);
  }
  auto res = GraphUtils::ArgsToAbstract(arg, converted, false);
  if (res->isa<mindspore::abstract::AbstractTensor>()) {
    bool check = CheckAdapterTensor(arg);
    dyn_cast_ptr<mindspore::abstract::AbstractTensor>(res)->set_is_adapter(check);
  }
  return res;
}

bool TensorInferBinarySupport(int opcode) {
  static const std::set<int> support_op = {
    BINARY_POWER,         BINARY_MULTIPLY,     BINARY_MODULO,       BINARY_ADD,
    BINARY_SUBTRACT,      BINARY_SUBSCR,       BINARY_FLOOR_DIVIDE, BINARY_TRUE_DIVIDE,
    INPLACE_FLOOR_DIVIDE, INPLACE_TRUE_DIVIDE, INPLACE_ADD,         INPLACE_SUBTRACT,
    INPLACE_MULTIPLY,     INPLACE_MODULO,      BINARY_LSHIFT,       BINARY_RSHIFT,
    BINARY_AND,           BINARY_XOR,          BINARY_OR,           INPLACE_POWER,
    INPLACE_LSHIFT,       INPLACE_RSHIFT,      INPLACE_AND,         INPLACE_XOR,
    INPLACE_OR,
  };

  return support_op.find(opcode) != support_op.end();
}

mindspore::abstract::AbstractTensorPtr InferWithMetaFunc(const AbstractBasePtr &left, const AbstractBasePtr &right,
                                                         int opcode) {
  auto func = GraphUtils::GetPrimOrMetaFuncGraph(opcode);
  auto res = mindspore::pipeline::AbstractAnalyze(GetValueNode(func), {left, right});
  return dyn_cast<mindspore::abstract::AbstractTensor>(res.eval_result->abstract());
}

mindspore::abstract::AbstractTensorPtr InferWithPrim(const AbstractBasePtr &left, const AbstractBasePtr &right,
                                                     int opcode) {
  static std::unordered_map<int, PrimitivePtr> prim_func = {{BINARY_ADD, prim::kPrimAdd},
                                                            {BINARY_SUBTRACT, prim::kPrimSub},
                                                            {BINARY_MULTIPLY, prim::kPrimMul},
                                                            {BINARY_TRUE_DIVIDE, prim::kPrimDiv},
                                                            {BINARY_FLOOR_DIVIDE, prim::kPrimFloorDiv}};

  auto left_dtype_ptr = dyn_cast_ptr<mindspore::abstract::AbstractTensor>(left)->element()->BuildType();
  MS_EXCEPTION_IF_NULL(left_dtype_ptr);
  auto right_dtype_ptr = dyn_cast_ptr<mindspore::abstract::AbstractTensor>(right)->element()->BuildType();
  MS_EXCEPTION_IF_NULL(right_dtype_ptr);
  if (left_dtype_ptr->type_id() != right_dtype_ptr->type_id() || prim_func.find(opcode) == prim_func.end()) {
    return InferWithMetaFunc(left, right, opcode);
  }

  auto func = prim_func.find(opcode)->second;
  auto infer_res = mindspore::abstract::TryInferAbstract(func, {left, right});

  if (infer_res.has_value()) {
    MS_EXCEPTION_IF_NULL(infer_res.value());
    return dyn_cast<mindspore::abstract::AbstractTensor>(infer_res.value());
  } else {
    return nullptr;
  }
}

py::object TensorInferBinary(const AbstractBasePtr &left, const AbstractBasePtr &right, int opcode) {
  mindspore::abstract::AbstractTensorPtr abs;
  auto left_tensor = dyn_cast_ptr<mindspore::abstract::AbstractTensor>(left);
  if (right->isa<mindspore::abstract::AbstractTensor>()) {
    abs = InferWithPrim(left, right, opcode);
  } else if (right->isa<mindspore::abstract::AbstractScalar>()) {
    auto new_right = std::make_shared<mindspore::abstract::AbstractTensor>(right);
    abs = InferWithPrim(left, new_right, opcode);
  } else {
    abs = InferWithMetaFunc(left, right, opcode);
  }
  MS_EXCEPTION_IF_NULL(abs);
  auto dtype_ptr = abs->element()->BuildType();
  MS_EXCEPTION_IF_NULL(dtype_ptr);
  auto shape_ptr = abs->BuildShape();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  auto shape = shape_ptr->cast<mindspore::abstract::ShapePtr>()->shape();
  auto dtype = dtype_ptr->type_id();
  auto tensor = std::make_shared<mindspore::tensor::Tensor>(dtype, shape);
  auto res = py::cast(tensor);
  tensor->set_adapter_flag(left_tensor->is_adapter());
  py::object func = Utils::GetModuleAttr("mindspore.common.api", "_convert_python_data", false, true);
  return func(res);
}

AObject *AbstractTensor::Binary(AObject *other, int op) {
  if (op == IS_OP) {
    PyTypeObject *b = other ? other->GetTypeObject() : nullptr;
    PyTypeObject *a = GetTypeObject();
    return a != b && b != nullptr ? Convert(Py_False) : MakeAObject(kTypeBool);
  }

  if (other == nullptr || GetPyObject().ptr() == nullptr || !TensorInferBinarySupport(op)) {
    return MakeAObject(kTypeTensor);
  }

  AbstractBasePtr left = PyObjectToAbstract(this->GetPyObject());
  AbstractBasePtr right;
  if (other->GetPyObject().ptr() == nullptr) {
    // if other is scalar with empty value, then transfer to AbstractScalar
    // else return any value
    switch (other->GetType()) {
      case kTypeBool:
        right = std::make_shared<mindspore::abstract::AbstractScalar>(kValueAny, kBool);
        break;
      case kTypeInt:
        right = std::make_shared<mindspore::abstract::AbstractScalar>(kValueAny, kInt32);
        break;
      case kTypeFloat:
        right = std::make_shared<mindspore::abstract::AbstractScalar>(kValueAny, kFloat32);
        break;
      default:
        return MakeAObject(kTypeAnyValue);
    }
  } else {
    right = PyObjectToAbstract(other->GetPyObject());
  }
  auto res = TensorInferBinary(left, right, op);
  return Convert(res);
}

AObject *AbstractTensor::GetItem(AObject *key) {
  PyObject *s = value_.ptr();
  PyObject *i = key ? key->GetPyObject().ptr() : nullptr;
  PyObject *t = nullptr;
  if (s != nullptr && i != nullptr) {
    // avoid Tensor as index and Tensor data sync
    t = PyObject_GetItem(s, i);
    CHECK_PYTHON_EXCEPTION(t);
  } else {
    return MakeAObject(kTypeAnyValue);
  }
  py::object py_t = py::reinterpret_steal<py::object>(t);
  if (CheckAdapterTensor(value_) && !CheckAdapterTensor(py_t)) {
    if (mindspore::IsStubTensor(py_t)) {
      mindspore::abstract::AbstractTensorPtr abs =
        dyn_cast<mindspore::abstract::AbstractTensor>(PyObjectToAbstract(py_t));
      auto dtype_ptr = abs->element()->BuildType();
      MS_EXCEPTION_IF_NULL(dtype_ptr);
      auto shape_ptr = abs->BuildShape();
      MS_EXCEPTION_IF_NULL(shape_ptr);
      auto shape = shape_ptr->cast<mindspore::abstract::ShapePtr>()->shape();
      auto dtype = dtype_ptr->type_id();
      auto tensor = std::make_shared<mindspore::tensor::Tensor>(dtype, shape);
      tensor->set_adapter_flag(true);
      py_t = py::cast(tensor);
    } else {
      auto tensor = py::cast<mindspore::tensor::TensorPtr>(t);
      tensor->set_adapter_flag(true);
    }
    py::object func = Utils::GetModuleAttr("mindspore.common.api", "_convert_python_data", false, true);
    py_t = func(py_t);
  }
  return Convert(py_t);
}

AObject *AbstractTensor::Unary(int op) const {
  switch (op) {
    case UNARY_POSITIVE:
      return const_cast<AbstractTensor *>(this);
    case UNARY_NEGATIVE:
    case UNARY_INVERT: {
      AbstractTensor *res = static_cast<AbstractTensor *>(MakeAObject(kTypeTensor));
      auto it = attrs_.find("shape");
      if (it != attrs_.end()) {
        res->attrs_["shape"] = it->second;
      }
      it = attrs_.find("dtype");
      if (it != attrs_.end()) {
        res->attrs_["dtype"] = it->second;
      }
      return res;
    }
    case UNARY_NOT: {
      auto it = attrs_.find("shape");
      if (it == attrs_.end() || it->second == nullptr) {
        return MakeAObject(kTypeTensor);
      }
      AObject *shape_info = it->second;
      PyObject *shape = shape_info->GetPyObject().ptr();
      Py_ssize_t ndim = PyTuple_GET_SIZE(shape);
      if (ndim == 0 || (ndim == 1 && PyLong_AS_LONG(PyTuple_GET_ITEM(shape, 0))) == 1) {
        return MakeAObject(kTypeBool);
      }
      return MakeAObject(kTypeAnyValue);
    }
    default:
      break;
  }
  return MakeAObject(kTypeAnyValue);
}

static const std::unordered_map<std::string, AObject::Type> tensor_attr_type = {
  // py Tensor property
  {"shape", AObject::kTypeTuple},
  {"dtype", AObject::kTypeMSDType},
  {"size", AObject::kTypeInt},
  {"itemsize", AObject::kTypeInt},
  {"nbytes", AObject::kTypeInt},
  {"strides", AObject::kTypeTuple},
  {"ndim", AObject::kTypeInt},
  {"has_init", AObject::kTypeBool},
  {"H", AObject::kTypeTensor},
  {"mH", AObject::kTypeTensor},
  {"T", AObject::kTypeTensor},
  {"mT", AObject::kTypeTensor},
  // cpp Tensor property
  {"_shape", AObject::kTypeTuple},
  {"_dtype", AObject::kTypeMSDType},
  {"_size", AObject::kTypeInt},
  {"_itemsize", AObject::kTypeInt},
  {"_nbytes", AObject::kTypeInt},
  {"_strides", AObject::kTypeTuple},
  {"init_flag", AObject::kTypeBool},
  {"adapter_flag", AObject::kTypeBool},
  {"param_info", AObject::kTypeAnyValue},
};

// return an uninitialized python tensor
static PyObject *GetUninitializedTensor() {
  static PyObject *tensor = nullptr;
  if (tensor != nullptr) {
    return tensor;
  }
  py::object py_cls = Utils::GetModuleAttr("mindspore", "Tensor", false, true);
  py::object cpp_cls = Utils::GetModuleAttr("mindspore._c_expression", "Tensor", false, true);
  py::object dtype = Utils::GetModuleAttr("mindspore", "int32", false, true);
  py::tuple shape;
  tensor = py_cls(cpp_cls(dtype, shape)).inc_ref().ptr();
  return tensor;
}

AbstractTensor::AbstractTensor(const py::object &o, bool is_stub) : AbstractObject(kTypeTensor, o), is_stub_(is_stub) {}

AObject *AbstractTensor::GetAttr(const std::string &name) {
  if (value_.ptr()) {
    return this->AbstractObject::GetAttr(name);
  }

  PyObject *tmp = GetUninitializedTensor();
  if (type_object_ != Py_TYPE(tmp)) {
    // tensor subclass or StubTensor and it's subclass
    // generic attribute
    AObject *attr = this->AbstractObjectBase::GetAttr(name);
    attrs_[name] = attr;
    return attr;
  }
  // get attribute for exact mindspore.Tensor,
  // not MetaTensor, not mindspore._c_expression.Tensor, not StubTensor

  // known @property attribute
  auto iter = tensor_attr_type.find(name);
  if (iter != tensor_attr_type.end()) {
    AObject *attr = MakeAObject(iter->second);
    if (iter->second == kTypeTuple) {
      static_cast<AbstractTuple *>(attr)->SetElementType(kTypeInt);
    }
    attrs_[name] = attr;
    return attr;
  }

  // know function attribute
  PyObject *op = PyObject_GetAttrString(tmp, name.c_str());
  AObject *attr = Convert(op);
  if (op == nullptr) {
    PyErr_Clear();
  } else {
    Py_DECREF(op);
  }
  if (attr->GetType() == kTypeBoundMethod) {
    attr->SetAttr("__self__", this);
    Py_INCREF(Py_None);
    Py_SETREF(PyMethod_GET_SELF(op), Py_None);
  } else {
    // not initialized attribute is not accept
    attr = MakeAObject(kTypeAnyValue);
  }
  attrs_[name] = attr;
  return attr;
}

std::string AbstractTensor::ToString() const {
  std::stringstream s;
  py::object dtype, shape;
  std::stringstream extra_info;
  if (value_.ptr()) {
    dtype = value_.attr("dtype");
    shape = value_.attr("shape");
    if (is_stub_) {
      extra_info << "stub_tensor ";
    }
    extra_info << "init=" << (CheckTensorDataInitialized(value_) ? "True" : "False");
  }
  s << this->AbstractObjectBase::ToString() << '\'' << std::string(py::str(dtype.ptr())) << ','
    << std::string(py::str(shape.ptr())) << "' " << extra_info.str() << ' ';
  return s.str();
}

}  // namespace pijit
}  // namespace mindspore
