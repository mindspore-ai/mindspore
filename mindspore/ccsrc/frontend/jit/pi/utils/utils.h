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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_UTILS_H
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_UTILS_H

#include <string>
#include <vector>
#include <utility>
#include <map>
#include <chrono>
#include <algorithm>
#include <unordered_map>
#include "pybind11/pybind11.h"
#include "ir/cell.h"
#include "frontend/jit/pi/utils/opcode_declare.h"
#include "frontend/jit/pi/python_adapter/py_code.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace pijit {

namespace py = pybind11;

constexpr auto kTwo = 2;
constexpr auto kThree = 3;
constexpr auto kFive = 5;

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
                                                             const py::object &kw, bool ret_vector_args = false);

  // alias python 'print(func); import dis; dis.dis(func)'
  static void DisFuncObject(PyObject *);
  // alias python 'print(...)'
  static void PyBuiltinPrint(PyObject *, bool flush = false);

  static PyObject *MixedPrecisionTypeToDType(MixedPrecisionType mixedPrecisionType);

  /// \brief Convert index and slice to {start, step, len, is_slice}
  ///
  /// \param subscr the index or slice
  /// \param size the size of the object being accessed
  ///
  /// \return Format : {start, step, len, is_slice}, if the subscript is valid
  ///                  {}, if the subscript is invalid
  static std::vector<Py_ssize_t> FormatSubscript(const py::object &subscr, Py_ssize_t size);
};

enum class LogCfg : uint8_t {
  kAll = 0,
  kTraceSource,
  kTraceBytecode,
  kGuard,
  kGraphBreak,
  kBytecode,
  kRecompiles,
  kRecompilesVerbose,
  kDynamic,
  kOthers,  // only for developer
  kLogMax
};

extern bool g_pijit_log_conf[static_cast<int>(LogCfg::kLogMax)];
extern const std::unordered_map<std::string, LogCfg> g_pijit_log_map;

inline std::string GetPiJitLogName(LogCfg cfg) {
  auto it =
    std::find_if(g_pijit_log_map.begin(), g_pijit_log_map.end(), [cfg](const auto &kv) { return kv.second == cfg; });
  return it == g_pijit_log_map.end() ? "" : it->first;
}

inline bool IsPiJitLogOn(LogCfg cfg) {
  constexpr int kAllIndex = static_cast<int>(LogCfg::kAll);
  return g_pijit_log_conf[kAllIndex] ? true : g_pijit_log_conf[static_cast<int>(cfg)];
}

#define PIJIT_DEBUG_LOG(module)                                                          \
  MSLOG_IF(mindspore::kDebug, IsPiJitLogOn(module), mindspore::NoExceptionType, nullptr) \
    << "[" << GetPiJitLogName(module) << "] "

/* use python format pattern */
template <typename... Args>
py::str PyStringFormat(std::string fmt, Args &&... args) {
  if (fmt.back() == '\n') {
    fmt.back() = ' ';
  }
  PyObject *py_format = PyUnicode_FromFormat(fmt.c_str(), std::forward<Args>(args)...);
  if (py_format != nullptr) {
    return py::reinterpret_steal<py::str>(py_format);
  }
  throw py::error_already_set();
}

/**
 * if the log string size is greater than logger size, print it to stderr
 * glog is limit log string size, default logger size is 30000
 */
#define GRAPH_JIT_LOG_F(fmt, ...)                                                                 \
  do {                                                                                            \
    std::string logger_helper;                                                                    \
    constexpr size_t log_min_size = 30000;                                                        \
    MSLOG_IF(mindspore::kDebug, true, mindspore::NoExceptionType, nullptr)                        \
      << std::endl                                                                                \
      << ((logger_helper = std::string(PyStringFormat(fmt, ##__VA_ARGS__))).size() < log_min_size \
            ? logger_helper                                                                       \
            : (((void)operator<<(std::cerr, logger_helper).operator<<(std::endl)), ""));          \
  } while (0)

#define PY_PRINTF(fmt, ...)                                          \
  do {                                                               \
    Utils::PyBuiltinPrint(PyStringFormat(fmt, ##__VA_ARGS__).ptr()); \
  } while (0)

#define PY_PRINTF_WITH_FLUSH(fmt, ...)                                     \
  do {                                                                     \
    Utils::PyBuiltinPrint(PyStringFormat(fmt, ##__VA_ARGS__).ptr(), true); \
  } while (0)

#define REPLACE_PY_MEMBER(member, o)     \
  do {                                   \
    PyObject *py_replace_tmp = (member); \
    Py_XINCREF(o);                       \
    (member) = (o);                      \
    Py_XDECREF(py_replace_tmp);          \
  } while (0)

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

/// \brief Change the jit-syntax-level to strict mode, and revert it back after the scope ends.
class JitSyntaxLevelScope {
 public:
  JitSyntaxLevelScope() {
    origin_jit_syntax_level_ = common::GetEnv("MS_DEV_JIT_SYNTAX_LEVEL");
    common::SetEnv("MS_DEV_JIT_SYNTAX_LEVEL", "0");
  }
  ~JitSyntaxLevelScope() { common::SetEnv("MS_DEV_JIT_SYNTAX_LEVEL", origin_jit_syntax_level_.c_str()); }

 private:
  std::string origin_jit_syntax_level_;
};

bool HasMutableOrConstAttr(PyObject *obj);
bool IsMutableObj(const py::object &obj);
bool CheckMutableOrNonConstAttr(PyObject *obj);
bool HasDynamicLength(PyObject *obj);
bool CheckDynamicLength(PyObject *obj);
bool CheckScalar(PyObject *obj);
bool CheckContainer(PyObject *obj);
bool IsTensorPyObject(PyObject *obj);
bool IsCTensorPyObject(PyObject *obj);
bool IsMsClass(PyObject *obj);
bool IsNumpyObject(PyObject *obj);
bool IsZipPyObject(PyTypeObject *obj);
bool IsNoGradEnterFunc(const py::object &handle);
bool IsNoGradExitFunc(const py::object &handle);
bool IsPartialFunc(const py::object &handle);
const char *GetFuncName(const py::object &handle);

py::object ConvertToMsTensor(const py::object &tensor);
py::object ConvertCppTensorToMsTensor(const py::object &tensor);

std::string GetTopModule(const py::object &o);
py::object GetPyCodeObject(const py::object &any, bool exact_func = false);
bool CheckConstPyObject(PyObject *cnst);

template <typename T>
class PackCallStackHelper {
 public:
  struct Packed {
    std::vector<T> args_;
    std::map<std::string, T> kw_;
  };

  explicit PackCallStackHelper(int opcode) : opcode_(opcode) {}
  auto &result() { return result_; }

  template <typename CastSequence, typename CastKeyWords>
  bool Pack(const std::vector<T> &stack_args, CastSequence to_seq, CastKeyWords to_map, PyObject *kw_names) {
    if (opcode_ == CALL_FUNCTION_EX) {
      result_.args_ = to_seq(stack_args[0]);
      result_.kw_ = stack_args.size() > 1 ? to_map(stack_args[1]) : std::map<std::string, T>();
      return true;
    }
    if (!Opcode(opcode_).IsCall()) {
      return false;
    }
    size_t size = stack_args.size();
    if (kw_names != nullptr) {
#if !IS_PYTHON_3_11_PLUS
      size--;
#endif
      size_t keys_size = PyTuple_GET_SIZE(kw_names);
      size = size - keys_size;
      for (size_t i = 0; i < keys_size; ++i) {
        result_.kw_[PyUnicode_AsUTF8(PyTuple_GET_ITEM(kw_names, i))] = stack_args[size + i];
      }
    }
    std::copy(stack_args.begin(), stack_args.begin() + size, std::back_inserter(result_.args_));
    return true;
  }

 private:
  Packed result_;
  int opcode_;
};

template <typename E>
class BindArgumentsHelper {
 public:
  struct BindArguments {
    /**
     * args contains kw-only arguments, ordered by code.co_varnames
     * e.g:
     * def func(a,/,b,*va,c,**kw_va):
     *    pass
     * func.__code__.co_varnames = ('a', 'b', 'c', 'va', 'kw_va')
     * args = [a, b, c, va, kw_va]
     */
    std::vector<E> args_;
    std::vector<E> va_;
    std::map<std::string, E> kw_va_;
  };
  explicit BindArgumentsHelper(PyCodeObject *co) : co_(co) {
    results_.args_.resize(co->co_argcount + co->co_kwonlyargcount);
  }
  auto &results() { return results_; }

 private:
  PyCodeObject *co_;
  BindArguments results_;

 public:
  // return false if arguments not match, not check missing arguments
  bool Bind(const std::vector<E> &args, const std::map<std::string, E> &kw = {}) {
    const int argc = args.size();
    PyCodeWrapper wrapper(co_);
    const int position_only = wrapper.PositionOnlyArgCount();
    const int position_argc = std::min(argc, co_->co_argcount);
    const int parameter_argc = results_.args_.size();

    for (int i = 0; i < position_argc; i++) {
      results_.args_[i] = args[i];
    }
    if (co_->co_flags & CO_VARARGS) {
      results_.va_ = {args.begin() + position_argc, args.end()};
    } else if (argc > co_->co_argcount) {
      MS_LOG(ERROR) << "takes " << co_->co_argcount << " positional arguments but " << argc << " was given";
      return false;
    }
    auto vars = PyCodeWrapper(co_).VarNames();
    PyObject **begin = &PyTuple_GET_ITEM(vars.ptr(), 0);
    PyObject **end = begin + parameter_argc;
    for (const auto &item : kw) {
      const auto &k = item.first;
      const auto &v = item.second;
      auto iter = std::find_if(begin, end, [&k](PyObject *o) { return k == PyUnicode_AsUTF8(o); });
      auto index = iter - begin;
      if (iter == end && (co_->co_flags & CO_VARKEYWORDS) == 0) {
        MS_LOG(ERROR) << "got an unexpected keyword argument '" << k << "'";
        return false;
      }
      if (iter != end && (index < position_only || results_.args_[index] != E())) {
        MS_LOG(ERROR) << (index < position_only ? "missing 1 required positional argument: '"
                                                : "got multiple values for argument '")
                      << k << "'";
        return false;
      }
      if (iter != end) {
        results_.args_[index] = v;
      } else {
        results_.kw_va_.insert(item);
      }
    }
    return true;
  }

  // check no defaults arguments
  bool CheckArguments(size_t start, size_t stop) {
    auto begin = results_.args_.begin() + start;
    auto end = results_.args_.begin() + stop;
    auto iter = std::find(begin, end, E());
    if (iter == end) {
      return true;
    }
    auto index = iter - results_.args_.begin();
    MS_LOG(ERROR) << "missing 1 required positional argument at " << index;
    return false;
  }

  // return false if missing arguments
  template <typename Converter>
  bool ApplyDefault(PyObject *defaults, PyObject *kw_defaults, Converter convert) {
    defaults = defaults ? defaults : Py_None;
    kw_defaults = kw_defaults ? kw_defaults : Py_None;
    bool is_valid_defaults = (defaults == Py_None || PyTuple_Check(defaults));
    bool is_valid_kw_defaults = (kw_defaults == Py_None || PyDict_Check(kw_defaults));
    MS_EXCEPTION_IF_CHECK_FAIL(is_valid_defaults && is_valid_kw_defaults, "error arguments");
    const size_t defaults_size = defaults == Py_None ? 0 : PyTuple_GET_SIZE(defaults);
    const size_t off = co_->co_argcount - defaults_size;
    MS_EXCEPTION_IF_CHECK_FAIL(defaults_size <= static_cast<size_t>(co_->co_argcount), "error function defaults");
    if (!CheckArguments(0, off)) {
      return false;
    }
    auto varnames = PyCodeWrapper(co_).VarNames().ptr();
    for (size_t i = off, argc = results_.args_.size(); i < argc; ++i) {
      if (results_.args_[i] != E()) {
        continue;
      } else if (i - off < defaults_size) {
        results_.args_[i] = convert(defaults, py::int_(i - off).ptr(), PyTuple_GET_ITEM(defaults, i - off));
        continue;
      }
      auto k = PyTuple_GET_ITEM(varnames, i);
      auto v = PyDict_GetItem(kw_defaults, k);
      if (v == nullptr) {
        MS_LOG(ERROR) << "missing 1 required keyword-only argument: '" << PyUnicode_AsUTF8(k) << "'";
        return false;
      }
      results_.args_[i] = convert(kw_defaults, k, v);
    }
    return true;
  }
};

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

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PI_UTILS_H
