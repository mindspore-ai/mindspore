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
#include "ir/tensor.h"
#include "ir/map_tensor.h"
#include "pipeline/jit/pi_jit/utils/utils.h"
#include "pipeline/jit/pi_jit/pydef.h"
#include "pybind11/pybind11.h"
#include "utils/log_adapter.h"
#include "pipeline/jit/pi_jit/utils/opcode_util.h"

namespace mindspore {
namespace jit {
namespace graph {

namespace py = pybind11;

static const char kMutableAttr[] = "__ms_mutable__";
static const char kConstArgAttr[] = "const_arg";
static const char kDynamicLengthAttr[] = "__ms_dynamic_len__";

std::string GetStopTraceReasonDesc(StopTraceReason res) {
  switch (res) {
#define STOP_TRACE_REASON_KIND(kind, description) \
  case k##kind: {                                 \
    return description;                           \
  }
#include "stop_trace_reason.def"
#undef STOP_TRACE_REASON_KIND
    default: {
      MS_EXCEPTION_IF_CHECK_FAIL(false, "Undefined STOP_TRACE_REASON");
      return "";
    }
  }
}

std::string GetInlineReasonDesc(InlineReason res) {
  switch (res) {
#define INLINE_REASON_KIND(kind, description) \
  case k##kind: {                             \
    return description;                       \
  }
#include "inline_reason.def"
#undef INLINE_REASON_KIND
    default: {
      MS_EXCEPTION_IF_CHECK_FAIL(false, "Undefined INLINE_REASON");
      return "";
    }
  }
}

std::string GetLoopUnrollingReasonDesc(LoopUnrollingReason res) {
  switch (res) {
#define LOOP_UNROLLING_REASON_KIND(kind, description) \
  case k##kind: {                                     \
    return description;                               \
  }
#include "loop_unrolling_reason.def"
#undef LOOP_UNROLLING_REASON_KIND
    default: {
      MS_EXCEPTION_IF_CHECK_FAIL(false, "Undefined LOOP_UNROLLING_REASON");
      return "";
    }
  }
}

std::string Utils::GetPyName(PyObject *obj) {
  const char *str = PyUnicode_AsUTF8(obj);
  return str != nullptr ? std::string(str) : "";
}

int Utils::GetBranchDestIndex(int op, int arg, int ci) {
  if (Utils::IsRelativeJump(op)) {
    return ci + 1 + arg / sizeof(_Py_CODEUNIT);
  }
  if (Utils::IsAbsoluteJump(op)) {
    return arg / sizeof(_Py_CODEUNIT);
  }
  return -1;
}

int Utils::GetBranchDestArg(int op, int jump_bci, int curr_bci) {
  if (Utils::IsRelativeJump(op)) {
    MS_EXCEPTION_IF_CHECK_FAIL(jump_bci > curr_bci, "invalid jump bci: " + std::to_string(jump_bci) +
                                                      " current bci: " + std::to_string(curr_bci));
    return (jump_bci - curr_bci - 1) * sizeof(_Py_CODEUNIT);
  }
  if (Utils::IsAbsoluteJump(op)) {
    return jump_bci * sizeof(_Py_CODEUNIT);
  }
  MS_EXCEPTION_IF_CHECK_FAIL(false, "undefined branch opcode: " + GetOpName(op));
  return -1;
}

bool Utils::IsRelativeJump(int op) { return code::GetOpcodeInfo(op).flag_ & code::kJRel; }
bool Utils::IsAbsoluteJump(int op) { return code::GetOpcodeInfo(op).flag_ & code::kJAbs; }
bool Utils::IsNameRelated(int op) { return code::GetOpcodeInfo(op).flag_ & code::kNamed; }
bool Utils::IsCallOp(int op) { return code::GetOpcodeInfo(op).flag_ & code::kCall; }
bool Utils::IsNonFall(int op) { return code::GetOpcodeInfo(op).flag_ & code::kNoFall; }
bool Utils::IsIfJump(int op) { return (code::GetOpcodeInfo(op).flag_ & (code::kJAbs | code::kNoFall)) == code::kJAbs; }
bool Utils::IsLocalAccessOp(int op) { return code::GetOpcodeInfo(op).flag_ & code::kLocalAccess; }
// it is PyCellObject access op, only used for cell/free/closure
bool Utils::IsCellAccessOp(int op) { return code::GetOpcodeInfo(op).flag_ & code::kCellAccess; }
bool Utils::IsGeneralNoSideEffectOp(int op) { return code::GetOpcodeInfo(op).flag_ & code::kGeneralNoSideEffect; }
bool Utils::IsNoSideEffectOp(int op) { return code::GetOpcodeInfo(op).flag_ & code::kNoSideEffect; }
bool Utils::IsLoadOp(int op) { return code::GetOpcodeInfo(op).flag_ & code::kLoad; }
bool Utils::IsMsUnsupported(int op) { return code::GetOpcodeInfo(op).flag_ & code::kMsUnsupported; }
bool Utils::IsBinaryMathOp(int op) { return code::GetOpcodeInfo(op).flag_ & code::kBinaryMath; }
const std::string &Utils::GetOpName(int op) { return code::GetOpcodeInfo(op).name_; }

void Utils::PyBuiltinPrint(PyObject *str) {
  static _PyCFunctionFastWithKeywords pyprint = nullptr;
  if (!pyprint) {
    PyObject *p = PyDict_GetItemString(PyEval_GetBuiltins(), "print");
    MS_EXCEPTION_IF_CHECK_FAIL(p && PyCFunction_Check(p), "can't get python 'print' function");
    pyprint = (_PyCFunctionFastWithKeywords)PyCFunction_GET_FUNCTION(p);
  }
  pyprint(nullptr, &str, 1, nullptr);
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
  auto dis = py::module::import("dis").attr("dis");
  // py::print("*** Dump ByteCode After CodeGen on [", py::cast<py::object>(func), "] ***");
  PY_PRINT_F("*** Dump ByteCode After CodeGen on [%A] ***", func);
  auto args = PyTuple_Pack(1, func);
  Py_XDECREF(PyObject_Call(dis.ptr(), args, NULL));
  Py_DECREF(args);
  if (PyErr_Occurred()) {
    PyErr_Print();
  }
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
  if (attr == nullptr) {
    if (_throw) {
      throw py::error_already_set();
    }
    Utils::ReportPythonException();
    PyErr_Clear();
  }
  return py::reinterpret_steal<py::object>(attr);
}

std::string Utils::ReportPythonException() {
  if (PyErr_Occurred()) {
    PyObject *et, *ev, *tb;
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
      PyObject *new_args = PySequence_Concat(pargs.ptr(), vals);
      pargs = py::reinterpret_steal<py::object>(new_args);
      kwargs = py::reinterpret_steal<py::object>(keys);
      Py_DECREF(vals);
    }
    break;
  } while (0);
  return {pargs, kwargs};
}

std::pair<py::object, py::object> Utils::PackCallStackArgs(const std::vector<py::object> &args, int callop,
                                                           bool ret_vector_args) {
  std::pair<py::object, py::object> failed;
  size_t args_size = args.size();
  if (std::find_if(args.begin(), args.end(), [](const py::object &o) { return o.ptr() == nullptr; }) != args.end()) {
    return failed;
  }
  Py_ssize_t psize = args_size;
  Py_ssize_t kwsize = 0;
  py::object pargs;
  py::object kwargs;
  switch (callop) {
    case CALL_FUNCTION_KW: {
      py::object keys = args.back();
      if (!PyTuple_Check(keys.ptr())) {
        return failed;
      }
      psize--;
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
      // handle tuple
    }
    /* fall-through */
    case CALL_FUNCTION:
    case CALL_METHOD:
      pargs = py::tuple(psize);
      for (Py_ssize_t i = 0; i < psize; ++i) {
        PyTuple_SET_ITEM(pargs.ptr(), i, args[i].inc_ref().ptr());
      }
      break;
    case CALL_FUNCTION_EX:
      return PackExArgs(args, ret_vector_args);
    default:
      return failed;
  }
  return {pargs, kwargs};
}

#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION < 9)
static PyObject *RetFrame(PyFrameObject *f, int) {
  Py_INCREF(f);
  return reinterpret_cast<PyObject *>(f);
}
#else
static PyObject *RetFrame(PyThreadState *, PyFrameObject *f, int) {
  Py_INCREF(f);
  return reinterpret_cast<PyObject *>(f);
}
#endif

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
  MS_LOG(DEBUG) << "prepare frame for " << std::string(py::str(callable)) << " failed because "
                << Utils::ReportPythonException();
  return nullptr;
}

bool HasMutableOrConstAttr(PyObject *obj) {
  auto pyObj = py::cast<py::object>(obj);
  return py::hasattr(pyObj, kMutableAttr) || py::hasattr(pyObj, kConstArgAttr);
}

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
  return PyLong_Check(obj) || PyFloat_Check(obj) || PyBool_Check(obj) || PyUnicode_Check(obj) || PyBytes_Check(obj) ||
         PyComplex_Check(obj);
}

bool CheckContainer(PyObject *obj) {
  return PyList_Check(obj) || PyTuple_Check(obj) || PySet_Check(obj) || PyDict_Check(obj);
}

bool IsTensorPyObject(PyObject *obj) {
  return py::isinstance<mindspore::tensor::MapTensor>(obj) || py::isinstance<mindspore::tensor::Tensor>(obj) ||
         py::isinstance<mindspore::tensor::MetaTensor>(obj) || py::isinstance<mindspore::tensor::CSRTensor>(obj) ||
         py::isinstance<mindspore::tensor::RowTensor>(obj) || py::isinstance<mindspore::tensor::COOTensor>(obj) ||
         py::isinstance<mindspore::tensor::TensorData>(obj);
}

}  // namespace graph
}  // namespace jit
}  // namespace mindspore
