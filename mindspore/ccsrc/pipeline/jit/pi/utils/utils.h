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
#ifndef MINDSPORE_PI_JIT_UTILS_H
#define MINDSPORE_PI_JIT_UTILS_H

#include <string>
#include <vector>
#include <utility>
#include <map>
#include <chrono>
#include "pybind11/pybind11.h"
#include "pipeline/jit/pi/pydef.h"
#include "mindspore/core/ir/cell.h"

namespace mindspore {
namespace pijit {

namespace py = pybind11;

constexpr auto kTwo = 2;
constexpr auto kThree = 3;
constexpr auto kFive = 5;

enum StopTraceReason : uint8_t {
#define STOP_TRACE_REASON_KIND(kind, description) k##kind,
#include "stop_trace_reason.def"
#undef STOP_TRACE_REASON_KIND
};

std::string GetStopTraceReasonDesc(StopTraceReason res);

enum InlineReason : uint8_t {
#define INLINE_REASON_KIND(kind, description) k##kind,
#include "inline_reason.def"
#undef INLINE_REASON_KIND
};

std::string GetInlineReasonDesc(InlineReason res);

enum LoopUnrollingReason : uint8_t {
#define LOOP_UNROLLING_REASON_KIND(kind, description) k##kind,
#include "loop_unrolling_reason.def"
#undef LOOP_UNROLLING_REASON_KIND
};

std::string GetLoopUnrollingReasonDesc(LoopUnrollingReason res);

class Utils {
 public:
  Utils() = default;
  ~Utils() = default;

  static std::string GetPyName(PyObject *obj);

  static PyFrameObject *PrepareFrame(PyObject *callable, PyObject *args, PyObject *kwargs);

  // find a object from specified module. default not import, not throw.
  static py::object GetModuleAttr(const std::string &mod_name, const std::string &attr_name, bool _import = false,
                                  bool _throw = false);

  // if has a python exception, log it and return the exception information
  static std::string ReportPythonException();

  /**
   * Pack stack arguments to PyObject by opcode
   *
   * \param args stack arguments, the layout match opcode.
   * \param callop CALL_FUNCTION/CALL_METHOD/CALL_FUNCTION_KW/CALL_FUNCTION_EX.
   * \param ret_vector_args if true, return a tuple arguments with names tuple.
   *                        default, return a tuple arguments with a dict arguments.
   *                        if failed, pair.first is empty.
   * \return a pair of arguments for object call
   */
  static std::pair<py::object, py::object> PackCallStackArgs(const std::vector<py::object> &args, int opcode,
                                                             bool ret_vector_args = false);

  // alias python 'print(func); import dis; dis.dis(func)'
  static void DisFuncObject(PyObject *);
  // alias python 'print(...)'
  static void PyBuiltinPrint(PyObject *);

  static PyObject *MixedPrecisionTypeToDType(MixedPrecisionType mixedPrecisionType);
};

#define GRAPH_JIT_LOG_F PY_PRINT_F

#define PY_PRINT_F(fmt, ...)                                       \
  do {                                                             \
    PyObject *_pystr;                                              \
    if (fmt[strlen(fmt) - 1] == '\n') {                            \
      std::string _fstr = fmt;                                     \
      _fstr[_fstr.size() - 1] = ' ';                               \
      _pystr = PyUnicode_FromFormat(_fstr.c_str(), ##__VA_ARGS__); \
    } else {                                                       \
      _pystr = PyUnicode_FromFormat(fmt, ##__VA_ARGS__);           \
    }                                                              \
    Utils::PyBuiltinPrint(_pystr);                                 \
    Py_DECREF(_pystr);                                             \
  } while (0)

#define REPLACE_PY_MEMBER(member, o)     \
  do {                                   \
    PyObject *py_replace_tmp = (member); \
    Py_XINCREF(o);                       \
    (member) = (o);                      \
    Py_XDECREF(py_replace_tmp);          \
  } while (0)

#ifdef DEBUG
#define PRINT_IF_HAS_USER_DEFINED_HOOK(op, hook)                                         \
  do {                                                                                   \
    static const char *slot_key_##hook = #hook;                                          \
    PyObject *attr_##hook = PyObject_GetAttrString(op, slot_key_##hook);                 \
    if (attr_##hook && (PyMethod_Check(attr_##hook) || PyFunction_Check(attr_##hook))) { \
      PY_PRINT_F("%A has hook " #hook, PyType_Check(op) ? op : (PyObject *)Py_TYPE(op)); \
    } else {                                                                             \
      PyErr_Clear();                                                                     \
    }                                                                                    \
    Py_XDECREF(attr_##hook);                                                             \
  } while (0)
#else
#define PRINT_IF_HAS_USER_DEFINED_HOOK(op, hook)
#endif
class ReprRecursionScope {
 public:
  explicit ReprRecursionScope(PyObject *v) : v_(v), stat_(v == nullptr ? -1 : Py_ReprEnter(v)) {}
  ~ReprRecursionScope() {
    if (stat_ == 0) {
      Py_ReprLeave(v_);
    }
  }
  bool ErrExist() { return stat_ < 0; }
  bool ReEnter() { return stat_ > 0; }
  bool ReEnterOrError() { return ReEnter() || ErrExist(); }

 private:
  PyObject *v_;
  int stat_;
};

bool HasMutableOrConstAttr(PyObject *obj);
bool IsMutableObj(const py::object &obj);
bool CheckMutableOrNonConstAttr(PyObject *obj);
bool HasDynamicLength(PyObject *obj);
bool CheckDynamicLength(PyObject *obj);
bool CheckScalar(PyObject *obj);
bool CheckContainer(PyObject *obj);
bool IsTensorPyObject(PyObject *obj);
bool IsMsClass(PyObject *obj);
bool IsNumpyObject(PyObject *obj);
const char *GetFuncName(const py::object &handle);

bool CheckAdapterTensor(const py::object &tensor);
py::object ConvertToMsTensor(const py::object &tensor);
py::object ConvertToAdapterTensor(const py::object &tensor);

std::string GetTopModule(const py::object &o);
py::object GetPyCodeObject(const py::object &any, bool exact_func = false);
size_t DeviceAvailableMemSize();
bool CheckConstPyObject(PyObject *cnst);

class TimeRecorder {
 public:
  using RecorderType = const char *;
  static constexpr double scale = std::nano::den;

  class TimeData {
   public:
    struct Data {
      uint64_t count;
      uint64_t nano;
    };
    TimeData() = default;
    ~TimeData();
    std::string ToString();

    std::map<RecorderType, Data> data_;
  };

  explicit TimeRecorder(const RecorderType &descr, bool record = true);
  ~TimeRecorder();

 private:
  static TimeData *Data();

  RecorderType descr_;
  std::chrono::steady_clock::time_point start_;
  bool record_;
};

class RefTracker {
 public:
  ~RefTracker();
  bool Track(PyObject *obj, const std::string &descr);
  static RefTracker *GetInstance();

 private:
  RefTracker();
  static PyObject *UnTrack(PyObject *ref, PyObject *);
  std::map<void *, std::pair<PyObject *, PyObject *>> tracked_;
  PyMethodDef mdef_;
};

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_UTILS_H
