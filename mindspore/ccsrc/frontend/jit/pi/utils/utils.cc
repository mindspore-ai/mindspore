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
#include "frontend/jit/pi/utils/utils.h"
#include <unordered_set>
#include <iomanip>
#include "ir/tensor.h"
#include "ir/tensor_data.h"
#include "ir/map_tensor.h"
#include "frontend/jit/pi/python_adapter/pydef.h"
#include "pybind11/pybind11.h"
#include "frontend/jit/pi/utils/opcode_util.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "frontend/jit/pi/utils/opcode_declare.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_info.h"
#include "include/utils/tensor_py.h"

namespace mindspore {
namespace pijit {

namespace py = pybind11;

static const char kMutableAttr[] = "__ms_mutable__";
static const char kConstArgAttr[] = "const_arg";
static const char kDynamicLengthAttr[] = "__ms_dynamic_len__";
static const char kMsClassAttr[] = "__ms_class__";
static const int DefaultPercision = 6;

std::string Utils::GetPyName(PyObject *obj) {
  const char *str = PyUnicode_AsUTF8(obj);
  return str != nullptr ? std::string(str) : "";
}

void Utils::PyBuiltinPrint(PyObject *str, bool flush) {
  static _PyCFunctionFastWithKeywords pyprint = nullptr;
  if (!pyprint) {
    PyObject *p = PyDict_GetItemString(PyEval_GetBuiltins(), "print");
    MS_EXCEPTION_IF_CHECK_FAIL(p && PyCFunction_Check(p), "can't get python 'print' function");
    pyprint = (_PyCFunctionFastWithKeywords)PyCFunction_GET_FUNCTION(p);
  }
  // The arguments of _PyCFunctionFastWithKeywords, refer to: https://docs.python.org/3/c-api/structures.html
  // This api has four arguments:
  // - PyObject *self, the `self` object of bounded method, or the module object.
  // - PyObject *const *args, an array of positional-args and keyword-args.
  // - Py_ssize_t nargs, the num of positional-args (not include keyword-args).
  // - PyObject *kwnames, a tuple of keyword-args names.
  if (flush) {
    PyObject *args[2]{str, Py_True};
    py::tuple kwnames = py::make_tuple("flush");
    pyprint(nullptr, args, 1, kwnames.ptr());
  } else {
    pyprint(nullptr, &str, 1, nullptr);
  }
  if (PyErr_Occurred()) {
    PyErr_Print();
    PyErr_Clear();
  }
}

void Utils::DisFuncObject(PyObject *func) {
  if (func == nullptr) {
    GRAPH_JIT_LOG_F("(nil)\n");
    return;
  }
  py::object dis = py::module::import("dis").attr("dis");
  py::object args = py::reinterpret_steal<py::object>(PyTuple_Pack(1, func));
  py::object kw;
#if IS_PYTHON_3_11_PLUS
  kw = py::dict();
  PyDict_SetItemString(kw.ptr(), "show_caches", Py_True);
  PyDict_SetItemString(kw.ptr(), "adaptive", Py_True);
#endif
  py::object res = py::reinterpret_steal<py::object>(PyObject_Call(dis.ptr(), args.ptr(), kw.ptr()));
  if (res.ptr() == nullptr) {
    PyErr_Print();
  }
  // By adding `print("", flush=True)`, the output of `dis` can be immediately printed out without being truncated
  // by `MS_LOG`.
  PyBuiltinPrint(py::str("").ptr(), true);
}

py::object Utils::GetModuleAttr(const std::string &mod_name, const std::string &attr_name, bool _import, bool _throw) {
  PyObject *name = PyUnicode_FromString(mod_name.c_str());
  PyObject *mod = _import ? PyImport_Import(name) : PyImport_GetModule(name);
  PyObject *attr = nullptr;
  Py_DECREF(name);
  if (mod != nullptr) {
    attr = PyObject_GetAttrString(mod, attr_name.c_str());
    Py_DECREF(mod);
  }
  if (attr != nullptr) {
    return py::reinterpret_steal<py::object>(attr);
  }
  if (!_throw) {
    PyErr_Clear();
    return py::object();
  }
  if (!PyErr_Occurred()) {
    if (mod == nullptr) {
      if (_import) {
        PyErr_Format(PyExc_ModuleNotFoundError, "No module named %s", mod_name.c_str());
      } else {
        PyErr_Format(PyExc_KeyError, "sys.modules[%s]", mod_name.c_str());
      }
    } else if (attr == nullptr) {
      PyErr_Format(PyExc_AttributeError, "%S no attribute %s", mod, attr_name.c_str());
    }
  }
  throw py::error_already_set();
}

std::string Utils::ReportPythonException() {
  if (PyErr_Occurred()) {
    PyObject *et;
    PyObject *ev;
    PyObject *tb;
    PyErr_Fetch(&et, &ev, &tb);
    py::object s = py::str(ev);
    MS_LOG(DEBUG) << "has python exception " << PyUnicode_AsUTF8(s.ptr());
    return PyUnicode_AsUTF8(s.ptr());
  }
  return std::string();
}

static std::pair<py::object, py::object> PackExArgs(const std::vector<py::object> &args, bool ret_vector_args) {
  std::pair<py::object, py::object> failed;
  py::object pargs;
  py::object kwargs;
  const int args_size = args.size();
  do {
    if (args_size == 0) {
      return failed;
    }
    if (PyTuple_Check(args[0].ptr())) {
      pargs = args[0];
    } else {
      pargs = py::reinterpret_steal<py::object>(PySequence_Tuple(args[0].ptr()));
      PyErr_Clear();
    }
    if (args_size == 1 || pargs.ptr() == nullptr) {
      break;
    }
    if (PyDict_Check(args[1].ptr())) {
      kwargs = args[1];
    } else {
      kwargs = py::dict();
      PyDict_Update(kwargs.ptr(), args[1].ptr());
    }
    if (PyErr_Occurred()) {
      PyErr_Clear();
      return failed;
    }
    if (ret_vector_args) {
      PyObject *vals = PyDict_Values(kwargs.ptr());
      PyObject *keys = PyDict_Keys(kwargs.ptr());
      PyObject *keys_tuple = PySequence_Tuple(keys);
      PyObject *vals_tuple = PySequence_Tuple(vals);
      PyObject *new_args = PySequence_Concat(pargs.ptr(), vals_tuple);
      Py_DECREF(keys);
      Py_DECREF(vals);
      Py_DECREF(vals_tuple);
      pargs = py::reinterpret_steal<py::tuple>(new_args);
      kwargs = py::reinterpret_steal<py::tuple>(keys_tuple);
    }
  } while (0);
  return {pargs, kwargs};
}

std::pair<py::object, py::object> Utils::PackCallStackArgs(const std::vector<py::object> &args, int callop,
                                                           const py::object &kw_names, bool ret_vector_args) {
  std::pair<py::object, py::object> failed;
  size_t args_size = args.size();
  if (std::find_if(args.begin(), args.end(), [](const py::object &o) { return o.ptr() == nullptr; }) != args.end()) {
    return failed;
  }
  Py_ssize_t psize = args_size;
  Py_ssize_t kwsize = 0;
  py::object pargs;
  py::object kwargs;
  if (callop == CALL_FUNCTION_KW || callop == CALL_FUNCTION || callop == CALL || callop == CALL_METHOD) {
    if (kw_names.ptr() != nullptr) {
      py::object keys = kw_names;
      if (!PyTuple_Check(keys.ptr())) {
        return failed;
      }
#if !IS_PYTHON_3_11_PLUS
      psize--;
#endif
      kwsize = PyTuple_GET_SIZE(keys.ptr());
      if (psize < kwsize) {
        return failed;
      }
      if (ret_vector_args) {
        kwargs = keys;
      } else {
        psize -= kwsize;
        kwargs = kwsize ? py::dict() : py::object();
        for (auto i = 0; i < kwsize; ++i) {
          PyDict_SetItem(kwargs.ptr(), PyTuple_GET_ITEM(keys.ptr(), i), args[psize + i].ptr());
        }
      }
    }
    pargs = py::tuple(psize);
    for (Py_ssize_t i = 0; i < psize; ++i) {
      PyTuple_SET_ITEM(pargs.ptr(), i, args[i].inc_ref().ptr());
    }
  } else if (callop == CALL_FUNCTION_EX) {
    return PackExArgs(args, ret_vector_args);
  } else {
    return failed;
  }
  return {pargs, kwargs};
}

PyObject *RetFrame(PY_FRAME_EVAL_FUNCTION_SIGNATURE) {
  if (std::is_same<std::remove_pointer<decltype(f)>::type, PyFrameObject>::value) {
    Py_INCREF(f);
    return reinterpret_cast<PyObject *>(f);
  }
  MS_LOG(ERROR) << "python3.11 not implement prepare frame";
  return nullptr;
}

PyFrameObject *Utils::PrepareFrame(PyObject *callable, PyObject *args, PyObject *kwargs) {
  if (callable == nullptr || args == nullptr) {
    return nullptr;
  }
  PyInterpreterState *inter = PyInterpreterState_Main();
  _PyFrameEvalFunction prev = _PyInterpreterState_GetEvalFrameFunc(inter);
  _PyInterpreterState_SetEvalFrameFunc(inter, RetFrame);
  PyObject *frame = PyObject_Call(callable, args, kwargs);
  _PyInterpreterState_SetEvalFrameFunc(inter, prev);
  if (frame != nullptr) {
    return reinterpret_cast<PyFrameObject *>(frame);
  }
  return nullptr;
}

PyObject *Utils::MixedPrecisionTypeToDType(MixedPrecisionType mixed_type) {
  auto ms_dtype_obj = Utils::GetModuleAttr("mindspore", "dtype");
  auto dtype_fp16_obj = ms_dtype_obj.attr("float16").ptr();
  auto dtype_fp32_obj = ms_dtype_obj.attr("float32").ptr();
  auto dtype_bf16_obj = ms_dtype_obj.attr("bfloat16").ptr();
  auto dst_dtype = dtype_fp16_obj;
  if (mixed_type == MixedPrecisionType::kFP32) {
    dst_dtype = dtype_fp32_obj;
  } else if (mixed_type == MixedPrecisionType::kBF16) {
    dst_dtype = dtype_bf16_obj;
  }
  return dst_dtype;
}

std::vector<Py_ssize_t> Utils::FormatSubscript(const py::object &subscr, Py_ssize_t size) {
  if (subscr.ptr() == nullptr || subscr.is_none()) {
    return {};
  }
  if (PyIndex_Check(subscr.ptr())) {
    Py_ssize_t index = PyNumber_AsSsize_t(subscr.ptr(), NULL);
    if (index >= -size && index < size) {
      return {(index + size) % size, 1, 1, 0};
    }
  }
  if (PySlice_Check(subscr.ptr())) {
    Py_ssize_t start = 0;
    Py_ssize_t stop = 0;
    Py_ssize_t step = 0;
    if (PySlice_Unpack(subscr.ptr(), &start, &stop, &step) == 0) {
      auto len = PySlice_AdjustIndices(size, &start, &stop, step);
      return {start, step, (len < 0 ? 0 : len), 1};
    }
  }
  PyErr_Clear();
  return {};
}

bool HasMutableOrConstAttr(PyObject *obj) {
  auto pyObj = py::cast<py::object>(obj);
  return py::hasattr(pyObj, kMutableAttr) || py::hasattr(pyObj, kConstArgAttr);
}

bool IsMutableObj(const py::object &obj) { return py::getattr(obj, kMutableAttr, nullptr).ptr() == Py_True; }

bool CheckMutableOrNonConstAttr(PyObject *obj) {
  auto pyObj = py::cast<py::object>(obj);
  if (py::hasattr(pyObj, kMutableAttr)) {
    return py::cast<bool>(py::getattr(pyObj, kMutableAttr));
  } else if (py::hasattr(pyObj, kConstArgAttr)) {
    return !py::cast<bool>(py::getattr(pyObj, kConstArgAttr));
  } else {
    return false;
  }
}

bool HasDynamicLength(PyObject *obj) {
  auto pyObj = py::cast<py::object>(obj);
  return py::hasattr(pyObj, kDynamicLengthAttr);
}

bool CheckDynamicLength(PyObject *obj) {
  auto pyObj = py::cast<py::object>(obj);
  if (py::hasattr(pyObj, kDynamicLengthAttr) && py::cast<bool>(py::getattr(pyObj, kDynamicLengthAttr))) {
    return true;
  } else {
    return false;
  }
}

bool CheckScalar(PyObject *obj) {
  return PyLong_CheckExact(obj) || PyFloat_CheckExact(obj) || PyBool_Check(obj) || PyUnicode_CheckExact(obj) ||
         PyBytes_CheckExact(obj) || PyComplex_CheckExact(obj);
}

bool CheckContainer(PyObject *obj) {
  return PyList_CheckExact(obj) || PyTuple_CheckExact(obj) || PyAnySet_Check(obj) || PyDict_CheckExact(obj);
}

bool IsTensorPyObject(PyObject *obj) {
  return py::isinstance<mindspore::tensor::MapTensor>(obj) || mindspore::tensor::IsTensorPy(obj) ||
         py::isinstance<mindspore::tensor::CSRTensor>(obj) || py::isinstance<mindspore::tensor::RowTensor>(obj) ||
         py::isinstance<mindspore::tensor::COOTensor>(obj) || py::isinstance<mindspore::tensor::TensorData>(obj);
}

bool IsCTensorPyObject(PyObject *obj) {
  if (obj == nullptr) {
    return false;
  }
  py::handle mapped_type = py::detail::get_type_handle(typeid(mindspore::tensor::Tensor), false);
  PyTypeObject *tar = reinterpret_cast<PyTypeObject *>(mapped_type.ptr());
  if (tar == nullptr) {
    return false;
  }
  return Py_TYPE(obj) == tar;
}

bool IsMsClass(PyObject *obj) {
  if (obj == nullptr) {
    return false;
  }
  auto py_obj = py::cast<py::object>(obj);
  return py::hasattr(py_obj, kMsClassAttr) && py::cast<bool>(py::getattr(py_obj, kMsClassAttr));
}

bool IsNumpyObject(PyObject *op) {
  if (op == nullptr) {
    return false;
  }
  PyTypeObject *tp = Py_TYPE(op);
  constexpr const char numpy[] = "numpy";
  return tp->tp_name ? strncmp(tp->tp_name, numpy, sizeof(numpy) - 1) == 0 : false;
}

bool IsZipPyObject(PyTypeObject *obj) {
  if (obj == nullptr) {
    return false;
  }
  return obj == &PyZip_Type;
}

bool IsNoGradEnterFunc(const py::object &handle) {
  return std::string(py::str(handle)).find("_no_grad.__enter__") != std::string::npos;
}

bool IsNoGradExitFunc(const py::object &handle) {
  return std::string(py::str(handle)).find("_no_grad.__exit__") != std::string::npos;
}

bool IsPartialFunc(const py::object &handle) {
  return std::string(py::str(handle)).find("functools.partial") != std::string::npos;
}

std::string GetTopModule(const py::object &o) {
  PyObject *mod = PyObject_GetAttrString(o.ptr(), "__module__");
  const char *module_name = "";
  if (mod == nullptr) {
    PyErr_Clear();
  } else if (PyModule_Check(mod)) {
    module_name = PyModule_GetName(mod);
  } else if (PyUnicode_Check(mod)) {
    module_name = PyUnicode_AsUTF8(mod);
  }
  const char *s = strchr(module_name, '.');
  std::string res = s ? std::string(module_name, s - module_name) : module_name;
  Py_XDECREF(mod);
  return res;
}

py::object GetPyCodeObject(const py::object &any, bool exact_func) {
  PyObject *f = any.ptr();
  if (f == nullptr) {
    return py::object();
  }
  if (PyInstanceMethod_Check(f)) {
    f = PyInstanceMethod_GET_FUNCTION(f);
  }
  if (PyMethod_Check(f)) {
    f = PyMethod_GET_FUNCTION(f);
  }
  if (PyFunction_Check(f)) {
    f = PyFunction_GET_CODE(f);
  }
  if (PyCode_Check(f)) {
    return py::cast<py::object>(f);
  }
  if (exact_func) {
    return py::object();
  }
  PyObject *call = PyObject_GetAttrString(any.ptr(), "__call__");
  if (call == nullptr) {
    PyErr_Clear();
    return py::object();
  }
  // self reference self at __call__, recursive call self.
  // just call once
  return GetPyCodeObject(py::reinterpret_steal<py::object>(call), true);
}

const char *GetFuncName(const py::object &f) {
  PyObject *func = f.ptr();
  if (func == nullptr) {
    return "";
  }
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

bool CheckConstPyObject(PyObject *cnst) {
  static const std::unordered_set<PyTypeObject *> cnst_types = {
    Py_TYPE(Py_None), Py_TYPE(Py_Ellipsis), Py_TYPE(Py_True), &PyCode_Type,  &PyFloat_Type,
    &PyLong_Type,     &PyUnicode_Type,      &PyComplex_Type,  &PyBytes_Type,
  };
  if (cnst == nullptr) {
    return false;
  }
  if (PyTuple_CheckExact(cnst)) {
    // tuple can't reference self
    PyObject **begin = &PyTuple_GET_ITEM(cnst, 0);
    PyObject **end = begin + PyTuple_GET_SIZE(cnst);
    return end == std::find_if_not(begin, end, CheckConstPyObject);
  }
  return cnst_types.find(Py_TYPE(cnst)) != cnst_types.end();
}

py::object ConvertToMsTensor(const py::object &tensor) {
  py::object common_tensor_type = Utils::GetModuleAttr("mindspore", "Tensor", false, true);
  PyTypeObject *tp = reinterpret_cast<PyTypeObject *>(common_tensor_type.ptr());
  return Py_TYPE(tensor.ptr()) == tp ? tensor : common_tensor_type(tensor);
}

PyObject *GetMsTensorType();
py::object ConvertCppTensorToMsTensor(const py::object &any) {
  PyObject *op = any.ptr();
  py::handle mapped_type = py::detail::get_type_handle(typeid(mindspore::tensor::Tensor), false);
  PyTypeObject *cpp_tensor_type = reinterpret_cast<PyTypeObject *>(mapped_type.ptr());

  if (Py_IS_TYPE(op, cpp_tensor_type)) {
    py::object tp = py::reinterpret_borrow<py::object>(GetMsTensorType());
    return tp(any);
  }

  if (PyTuple_Check(op) || PyList_Check(op)) {
    for (Py_ssize_t i = 0; i < Py_SIZE(op); ++i) {
      PyObject **item = PyTuple_Check(op) ? &PyTuple_GET_ITEM(op, i) : &PyList_GET_ITEM(op, i);
      PyObject *new_item = ConvertCppTensorToMsTensor(py::cast<py::object>(*item)).inc_ref().ptr();
      Py_SETREF(*item, new_item);
    }
    return any;
  }

  if (PyDict_Check(op)) {
    Py_ssize_t pos = 0;
    PyObject *key;
    PyObject *value;
    while (PyDict_Next(op, &pos, &key, &value)) {
      py::object new_value = ConvertCppTensorToMsTensor(py::cast<py::object>(value));
      PyDict_SetItem(op, key, new_value.ptr());
    }
    return any;
  }
  return any;
}

RefTracker *RefTracker::GetInstance() {
  static RefTracker instance;
  return &instance;
}

PyObject *RefTracker::UnTrack(PyObject *self, PyObject *ref) {
  auto &refs = RefTracker::GetInstance()->tracked_;
  void *ptr = PyLong_AsVoidPtr(self);
  auto iter = refs.find(ptr);
  assert(iter != refs.end());
  Py_DECREF(iter->second.first);
  Py_DECREF(iter->second.second);
  refs.erase(iter);
  Py_RETURN_NONE;
}

bool RefTracker::Track(PyObject *obj, const std::string &descr) {
  if (obj == nullptr) {
    return false;
  }
  if (tracked_.find(obj) != tracked_.end()) {
    return true;
  }
  PyObject *self = PyLong_FromVoidPtr(obj);
  PyObject *callback = PyCFunction_New(&mdef_, self);
  PyObject *ref = PyWeakref_NewRef(obj, callback);
  Py_DECREF(callback);
  Py_DECREF(self);
  if (ref == nullptr) {
    PyErr_Clear();
    return false;
  }
  tracked_.insert({obj, {ref, PyUnicode_FromStringAndSize(descr.data(), descr.size())}});
  return true;
}

RefTracker::~RefTracker() {
  if (tracked_.empty()) {
    return;
  }
  std::cout << "ref tracker not empty" << std::endl;
  for (const auto &i : tracked_) {
    PyObject *obj = PyWeakref_GET_OBJECT(i.second.first);
    const char *descr = PyUnicode_AsUTF8(i.second.second);
    assert(obj == i.first && obj != nullptr);
    std::cout << "object " << (Py_TYPE(obj)->tp_name ? Py_TYPE(obj)->tp_name : "<unnamed>") << " at " << obj
              << " refcnt " << Py_REFCNT(obj) << " descr " << descr << std::endl;
  }
}

RefTracker::RefTracker() : mdef_({"ref_untrack", &RefTracker::UnTrack, METH_O, PyDoc_STR("")}) {}

TimeRecorder::TimeRecorder(const RecorderType &descr, bool record) : descr_(descr), record_(record) {
  if (record_) {
    start_ = std::chrono::steady_clock::now();
  }
}

TimeRecorder::~TimeRecorder() {
  if (record_) {
    uint64_t clk = static_cast<uint64_t>((std::chrono::steady_clock::now() - start_).count());
    auto &data = TimeRecorder::Data()->data_[descr_];
    data.count++;
    data.nano += clk;
  }
}

TimeRecorder::TimeData *TimeRecorder::Data() {
  static TimeData data;
  return &data;
}

TimeRecorder::TimeData::~TimeData() {
  if (!data_.empty()) {
    std::cout << ToString() << std::endl;
  }
}

std::string TimeRecorder::TimeData::ToString() {
  if (data_.empty()) {
    return std::string();
  }

  const auto Fmt = [](std::ostream &s) -> std::ostream & {
    const int w = 20;
    s << std::setw(w) << std::left;
    return s;
  };

  std::stringstream s;
  s.precision(DefaultPercision);
  s.setf(std::ios::fixed);
  s << "============= TimeRecorder =============" << std::endl;
  s << Fmt << "type" << Fmt << "times" << Fmt << "seconds" << std::endl;
  for (const auto &i : data_) {
    s << Fmt << i.first << Fmt << i.second.count << Fmt << (i.second.nano / TimeRecorder::scale) << std::endl;
  }
  s << "========================================" << std::endl;
  return s.str();
}

bool g_pijit_log_conf[static_cast<int>(LogCfg::kLogMax)] = {false};

const std::unordered_map<std::string, LogCfg> g_pijit_log_map = {
  {"all", LogCfg::kAll},
  {"trace_source", LogCfg::kTraceSource},
  {"trace_bytecode", LogCfg::kTraceBytecode},
  {"guard", LogCfg::kGuard},
  {"graph_break", LogCfg::kGraphBreak},
  {"bytecode", LogCfg::kBytecode},
  {"recompiles", LogCfg::kRecompiles},
  {"recompiles_verbose", LogCfg::kRecompilesVerbose},
  {"dynamic", LogCfg::kDynamic},
  {"others", LogCfg::kOthers},
};

}  // namespace pijit
}  // namespace mindspore
