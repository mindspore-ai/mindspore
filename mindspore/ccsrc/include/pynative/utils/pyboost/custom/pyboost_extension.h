/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_EXTENSION_PYBOOST_EXTENSION_H_
#define MINDSPORE_CCSRC_EXTENSION_PYBOOST_EXTENSION_H_

#include <vector>
#include <tuple>
#include <string>
#include <memory>
#include <utility>
#include <optional>
#include <type_traits>
#include "include/pynative/utils/pyboost/custom/tensor.h"
#include "mindspore/ccsrc/tools/profiler/profiler.h"
#include "include/pynative/forward/pyboost/fallback.h"
#include "include/utils/tensor_utils.h"
#include "include/pynative/utils/pynative_utils.h"
#include "include/utils/pynative/dispatch/ms_dispatch.h"
#include "include/utils/python_attr.h"
#include "include/utils/tensor_py.h"

namespace mindspore {
namespace stub {
class StubNode;
using StubNodePtr = std::shared_ptr<StubNode>;
}  // namespace stub
namespace device {
class DeviceContext;
}
namespace runtime {
class PyBoostDeviceTask;
}
}  // namespace mindspore

namespace ms {
namespace inner {
/**
 * @brief Retrieves the demangled function name (if applicable) from a mangled symbol name.
 * @param name A mangled symbol name.
 * @return The demangled function name (or the original name if demangling is not available).
 */
PYBOOST_API std::string GetFunctionName(const char *name);

/**
 * @brief Sets a promise for a stub node to associate it with a Tensor output.
 * @param op_name The name of the operation.
 * @param tuple A tuple of stub nodes.
 * @param output The Tensor output to associate with the promise.
 */
PYBOOST_API void SetPromise(const std::string &op_name, const std::tuple<mindspore::stub::StubNodePtr> &tuple,
                            const ms::Tensor &output);

/**
 * @brief Helper function for setting promises for multiple outputs.
 * @tparam Tuple The type of the tuple containing stub nodes.
 * @tparam Index The indices of the tuple elements.
 * @param op_name The name of the operation.
 * @param tuple A tuple of stub nodes.
 * @param outputs The Tensor outputs to associate with the promises.
 */
template <typename Tuple, size_t... Index>
void SetPromiseHelper(const std::string &op_name, const Tuple &tuple, const std::vector<ms::Tensor> &outputs,
                      std::index_sequence<Index...>) {
  (SetPromise(op_name, std::get<Index>(tuple), outputs[Index]), ...);
}

/**
 * @brief Sets promises for multiple outputs based on a tuple of stub nodes.
 * @tparam T The types of the elements in the tuple.
 * @param op_name The name of the operation.
 * @param tuple A tuple of stub nodes.
 * @param outputs The Tensor outputs to associate with the promises.
 * @throws If the tuple size does not match the size of the outputs.
 */
template <typename... T>
void SetPromise(const std::string &op_name, const std::tuple<T...> &tuple, const std::vector<ms::Tensor> &outputs) {
  constexpr size_t tuple_size = sizeof...(T);
  if (tuple_size != outputs.size()) {
    MS_LOG(EXCEPTION) << "For op " << op_name << ", the output size should be equal to " << tuple_size << ", but got "
                      << outputs.size();
  }
  SetPromiseHelper(op_name, tuple, outputs, std::make_index_sequence<tuple_size>{});
}

/**
 * @brief Generic converter entry point.
 * @tparam T Deduced type of the argument; must be one of
 *           Tensor, std::optional<Tensor>, or std::vector<U> where
 *           U is recursively any of the former.
 * @param arg Object whose innermost Tensor elements will be converted.
 */
template <typename T>
void ConvertMsTensor(const T &arg) {}

/**
 * @brief Converts a single Tensor.
 * @param arg Tensor instance to materialize.
 */
template <>
inline void ConvertMsTensor<ms::Tensor>(const ms::Tensor &arg) {
  arg.ConvertStubNodeToTensor();
}

/**
 * @brief Converts a Tensor wrapped in std::optional.
 * @param arg Optional tensor; conversion is skipped if disengaged.
 */
template <>
inline void ConvertMsTensor<std::optional<ms::Tensor>>(const std::optional<ms::Tensor> &arg) {
  if (arg.has_value()) {
    arg.value().ConvertStubNodeToTensor();
  }
}

/**
 * @brief Recursively converts every element of a vector.
 * @tparam T Value type stored in the vector; must itself be convertible
 *           via ConvertMsTensor.
 * @param vec Vector whose elements will be processed in order.
 */
template <typename T>
void ConvertMsTensor(const std::vector<T> &vec) {
  for (const auto &item : vec) {
    ConvertMsTensor(item);
  }
}

/**
 * @brief Converts multiple stub nodes into Tensor objects.
 * @tparam Args The types of the arguments.
 * @param args The arguments to convert.
 */
template <typename... Args>
void ConvertStubNodeToTensor(const Args &...args) {
  (ConvertMsTensor(args), ...);
}
}  // namespace inner

namespace pynative {
/**
 * @class PyboostRunner
 * @brief [API] Represents a runner for PyBoost operations, providing methods to manage execution, memory allocation,
 * and kernel launches.
 */
class PYBOOST_API PyboostRunner : public std::enable_shared_from_this<PyboostRunner> {
 public:
  /**
   * @brief Constructs a PyboostRunner with the specified operation name.
   * @param op_name The name of the operation.
   */
  explicit PyboostRunner(const std::string &op_name);

  /**
   * @brief Virtual destructor for PyboostRunner.
   */
  virtual ~PyboostRunner() = default;

  /**
   * @brief [API] Executes a given function and manages its output as PyBoost operation.
   * @tparam OUT_NUM The number of output tensors expected from the function.
   * @tparam FuncType The type of the function to be executed.
   * @tparam Args The types of the arguments to be passed to the function.
   * @param func The function to execute. It should return a result compatible with `SetPromise`.
   * @param args The arguments to pass to the function.
   * @return A Python object representing the outputs of the operation.
   *
   * @details This method:
   * - Records profiling information for the operation.
   * - Converts `StubNode` inputs to `Tensor` objects.
   * - Executes the provided function with the given arguments.
   * - Sets promises for the output tensors.
   * - Handles exceptions via a fallback mechanism.
   *
   * @throws If the function execution or promise setting fails, the exception is propagated.
   */
  template <int OUT_NUM, typename FuncType, typename... Args>
  static typename std::enable_if<(OUT_NUM > 0), py::object>::type Call(const std::string &op_name, FuncType func,
                                                                       Args &&...args) {
    mindspore::runtime::ProfilerRecorder profiler(mindspore::runtime::ProfilerModule::kPynative,
                                                  mindspore::runtime::ProfilerEvent::kRunOp, op_name);
    auto py_output = mindspore::tensor::MakeTuple<mindspore::tensor::TensorWrapper, OUT_NUM, false>();
    auto promises = mindspore::tensor::TransformPromise(py_output);
    mindspore::pynative::DispatchOp(std::make_shared<mindspore::pynative::PassthroughFrontendTask>(
      [=]() {
        inner::ConvertStubNodeToTensor(args...);
        auto result = func(args...);
        inner::SetPromise(op_name, promises, result);
      },
      [promises]() { mindspore::tensor::SetException(promises); }));
    return py::reinterpret_steal<py::object>(mindspore::tensor::TransformOutput(py_output));
  }

  template <int OUT_NUM, typename FuncType, typename... Args>
  static typename std::enable_if<(OUT_NUM > 0), py::object>::type Call(FuncType func, Args &&...args) {
    auto op_name = inner::GetFunctionName(typeid(FuncType).name());
    return Call<OUT_NUM>(op_name, func, std::forward<Args>(args)...);
  }

  template <int OUT_NUM, typename FuncType, typename... Args>
  static typename std::enable_if<(OUT_NUM == 0), py::object>::type Call(const std::string &op_name, FuncType func,
                                                                        Args &&...args) {
    mindspore::runtime::ProfilerRecorder profiler(mindspore::runtime::ProfilerModule::kPynative,
                                                  mindspore::runtime::ProfilerEvent::kRunOp, op_name);
    mindspore::pynative::DispatchOp(std::make_shared<mindspore::pynative::PassthroughFrontendTask>([=]() {
      inner::ConvertStubNodeToTensor(args...);
      func(args...);  // Call the function without return value
    }));
    return py::none();
  }

  template <int OUT_NUM, typename FuncType, typename... Args>
  static typename std::enable_if<(OUT_NUM == 0), py::object>::type Call(FuncType func, Args &&...args) {
    auto op_name = inner::GetFunctionName(typeid(FuncType).name());
    return Call<OUT_NUM>(op_name, func, std::forward<Args>(args)...);
  }

  /**
   * @brief [API] Executes the PyBoost operation with the given inputs and outputs.
   * @param inputs The input tensors.
   * @param outputs The output tensors.
   */
  void Run(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs);

  /**
   * @brief [API] Calculates the workspace size required for the operation.
   * @return The size of the workspace in bytes. Default is 0.
   */
  virtual size_t CalcWorkspace() { return 0; }

  /**
   * @brief Process after allocating workspace.
   */
  virtual void ProcessWithWorkspace() {}

  /**
   * @brief [API] Launches the kernel for the operation.
   */
  virtual void LaunchKernel() = 0;

  /**
   * @brief [API] Retrieves the operation name associated with this runner.
   * @return The operation name as a string.
   */
  const std::string &op_name() const { return _op_name_; }

  /**
   * @brief [API] Retrieves the input tensors.
   * @return A reference to the vector of input tensors.
   */
  const std::vector<ms::Tensor> &inputs() const { return _inputs_; }

  /**
   * @brief [API] Retrieves the output tensors.
   * @return A reference to the vector of output tensors.
   */
  const std::vector<ms::Tensor> &outputs() const { return _outputs_; }

  /**
   * @brief [API] Retrieves the stream ID associated with this runner.
   * @return The stream ID as an unsigned integer.
   */
  uint32_t stream_id() const { return _stream_id_; }

  /**
   * @brief [API] Retrieves the stream associated with this runner.
   * @return A pointer to the stream.
   */
  void *stream() { return _stream_; }

  /**
   * @brief [API] Retrieves the workspace pointer for the operation.
   * @return A pointer to the workspace memory.
   */
  void *workspace_ptr() { return _workspace_ptr_; }

 protected:
  /**
   * @brief Run with pyboost pipeline
   */
  virtual void _Run();

  /**
   * @brief Set stream id and stream ptr of pyboost runner.
   */
  virtual void _PrepareStream();

  /**
   * @brief Prepares device addresses for input and output tensors.
   */
  virtual void _PrepareDeviceAddress();

  /**
   * @brief Allocates tensors memory for inputs and outputs.
   */
  virtual void _MallocDeviceAddress();

  /**
   * @brief Allocates workspace memory.
   */
  virtual void _MallocWorkspace();

  /**
   * @brief Dispatch a launch task
   */
  virtual void _DispatchLaunchTask();

  /**
   * @brief Process cross stream address
   */
  void ProcessCrossStreamAddress();

  // The name of the operation.
  std::string _op_name_;
  // The input tensors.
  std::vector<ms::Tensor> _inputs_;
  // The output tensors.
  std::vector<ms::Tensor> _outputs_;
  // The stream ID for the operation.
  size_t _stream_id_{0};
  // The stream associated with the operation.
  void *_stream_{nullptr};
  // The device context for the operation.
  mindspore::device::DeviceContext *_device_context_{nullptr};
  // Pointer to the workspace memory.
  void *_workspace_ptr_{nullptr};
};

template <int OUT_NUM, typename Ret, typename... Args, size_t... I>
py::object PyboostArgsCallerNoDispatchImpl(Ret (*func)(Args...), const std::string &op_name, const py::args &args,
                                           std::index_sequence<I...>) {
  return ms::pynative::PyboostRunner::Call<OUT_NUM>(op_name, func, args[I].cast<std::decay_t<Args>>()...);
}

template <int OUT_NUM, typename Ret, typename... Args>
py::object PyboostArgsCallerNoDispatch(Ret (*func)(Args...), const std::string &op_name, const py::args &args) {
  constexpr size_t N = sizeof...(Args);
  if (args.size() != N) {
    MS_LOG(EXCEPTION) << "Argument count mismatch: expected " << N << ", got " << args.size();
  }
  return PyboostArgsCallerNoDispatchImpl<OUT_NUM>(func, op_name, args, std::make_index_sequence<N>());
}

inline py::object CustomGetFallbackFromPyObject(PyObject *obj) {
  if (obj == nullptr || mindspore::tensor::IsPyObjectPythonTensorStrict(obj) ||
      !mindspore::tensor::IsPyObjectTensorPy(obj)) {
    return {};
  }
  auto fallback = mindspore::FastGetPyObjectAttr(obj, mindspore::pynative::GetFallbackStr().c_str());
  if (fallback.ptr() != nullptr) {
    MS_LOG(DEBUG) << "[CustomPyBoost] Found __fallback__ from " << Py_TYPE(obj)->tp_name;
  }
  return fallback;
}

inline py::object CustomGetFallbackFromPyObjectOrSequence(PyObject *obj) {
  if (obj == nullptr) {
    return {};
  }
  if (PyTuple_Check(obj)) {
    Py_ssize_t size = PyTuple_Size(obj);
    for (Py_ssize_t i = 0; i < size; ++i) {
      PyObject *item = PyTuple_GetItem(obj, i);
      auto fallback = CustomGetFallbackFromPyObjectOrSequence(item);
      if (fallback.ptr() != nullptr) {
        MS_LOG(DEBUG) << "[CustomPyBoost] Found __fallback__ in tuple index " << i
                      << ", item type: " << Py_TYPE(item)->tp_name;
        return fallback;
      }
    }
    return {};
  }
  if (PyList_Check(obj)) {
    Py_ssize_t size = PyList_Size(obj);
    for (Py_ssize_t i = 0; i < size; ++i) {
      PyObject *item = PyList_GetItem(obj, i);
      auto fallback = CustomGetFallbackFromPyObjectOrSequence(item);
      if (fallback.ptr() != nullptr) {
        MS_LOG(DEBUG) << "[CustomPyBoost] Found __fallback__ in list index " << i
                      << ", item type: " << Py_TYPE(item)->tp_name;
        return fallback;
      }
    }
    return {};
  }
  return CustomGetFallbackFromPyObject(obj);
}

inline py::object CustomHandleFallback(const std::string &op_name, const py::args &args, const py::object &fallback,
                                       const py::object &callable) {
  MS_LOG(DEBUG) << "[CustomPyBoost] Enter fallback for op " << op_name << ", args size: " << args.size();
  PyObject *ret = PyObject_CallFunctionObjArgs(fallback.ptr(), callable.ptr(), args.ptr(), NULL);
  if (ret == nullptr) {
    if (PyErr_Occurred() != nullptr) {
      MS_LOG(DEBUG) << "[CustomPyBoost] Python fallback raised for op " << op_name;
      throw py::error_already_set();
    }
    MS_LOG(EXCEPTION) << "[CustomPyBoost] Python fallback failed for op " << op_name;
  }
  MS_LOG(DEBUG) << "[CustomPyBoost] Finish fallback for op " << op_name;
  return py::reinterpret_steal<py::object>(ret);
}

// CustomOpBuilder extensions can load their own copy of pybind type metadata, so create Python wrapper objects through
// exported MindSpore runtime helpers instead of casting header-defined C++ classes in the extension.
template <typename CppCallbackWrapper>
py::object CustomInvokeGuardChain(CppCallbackWrapper cpp_callback, const py::args &args, const std::string &op_name) {
  auto &stack = mindspore::pynative::DispatchGuardStack::Get();
  if (stack.NoMoreGuard()) {
    MS_LOG(DEBUG) << "[CustomPyBoost] Guard chain complete, executing original op: " << op_name;
    return cpp_callback(args);
  }

  const py::object &current_guard = stack.GetCurrentGuardAndMoveNext();
  py::object handler =
    mindspore::FastGetPyObjectAttr(current_guard.ptr(), mindspore::pynative::GetDispatchGuardCallbackStr().c_str());
  if (handler.ptr() == nullptr) {
    MS_LOG(EXCEPTION) << "Guard does not have valid " << mindspore::pynative::GetDispatchGuardCallbackStr()
                      << " attribute";
  }

  auto op_func = mindspore::pynative::MakeOpFunc(
    op_name, py::cpp_function([cpp_callback, op_name](const py::args &inner_args) mutable {
      return CustomInvokeGuardChain(cpp_callback, inner_args, op_name);
    }));
  MS_LOG(DEBUG) << "[CustomPyBoost] Invoke guard ms_dispatch for op " << op_name;
  return handler(op_func, args);
}

template <typename CppCallbackWrapper>
py::object CustomInvokeMsDispatch(const py::object &handler, CppCallbackWrapper cpp_callback, const py::args &args,
                                  const std::string &op_name) {
  auto op_func = mindspore::pynative::MakeOpFunc(
    op_name,
    py::cpp_function([cpp_callback, op_name](const py::args &inner_args) mutable { return cpp_callback(inner_args); }));
  MS_LOG(DEBUG) << "[CustomPyBoost] Invoke tensor-level ms_dispatch for op " << op_name;
  return handler(op_func, args, py::none());
}

template <int OUT_NUM, typename Ret, typename... Args>
py::object PyboostArgsCaller(Ret (*func)(Args...), const std::string &op_name, const py::args &args) {
  constexpr size_t N = sizeof...(Args);
  if (args.size() != N) {
    MS_LOG(EXCEPTION) << "Argument count mismatch: expected " << N << ", got " << args.size();
  }

  MS_LOG(DEBUG) << "[CustomPyBoost] Start op " << op_name << ", args size: " << args.size();
  const bool fallback_enabled = mindspore::pynative::fallback_enabled();
  auto fallback = fallback_enabled ? CustomGetFallbackFromPyObjectOrSequence(args.ptr()) : py::object();
  if (fallback.ptr() != nullptr) {
    MS_LOG(DEBUG) << "[CustomPyBoost] Route op " << op_name << " to __fallback__ before ms_dispatch";
    auto op_call_obj = mindspore::pynative::MakeOpCall(
      op_name, py::cpp_function([func, op_name](const py::args &inner_args, const py::kwargs &) {
        return PyboostArgsCaller<OUT_NUM>(func, op_name, inner_args);
      }));
    return CustomHandleFallback(op_name, args, fallback, op_call_obj);
  }
  MS_LOG(DEBUG) << "[CustomPyBoost] No __fallback__ route for op " << op_name
                << ", fallback enabled: " << fallback_enabled;

  if (mindspore::pynative::ms_dispatch_enabled()) {
    auto [dispatch_callback, is_guard_callback] = mindspore::pynative::GetDispatchCallbackFromInput(args);
    auto cpp_callback = [func, op_name](const py::args &inner_args) -> py::object {
      return PyboostArgsCaller<OUT_NUM>(func, op_name, inner_args);
    };
    if (is_guard_callback) {
      MS_LOG(DEBUG) << "[CustomPyBoost] Route op " << op_name << " to guard-chain ms_dispatch";
      mindspore::pynative::GuardChainScope guard_chain_scope;
      return CustomInvokeGuardChain(cpp_callback, args, op_name);
    }
    if (!dispatch_callback.is_none()) {
      MS_LOG(DEBUG) << "[CustomPyBoost] Route op " << op_name << " to tensor-level ms_dispatch";
      return CustomInvokeMsDispatch(dispatch_callback, cpp_callback, args, op_name);
    }
    MS_LOG(DEBUG) << "[CustomPyBoost] No ms_dispatch callback found for op " << op_name;
  } else {
    MS_LOG(DEBUG) << "[CustomPyBoost] ms_dispatch disabled for op " << op_name;
  }

  MS_LOG(DEBUG) << "[CustomPyBoost] Run op " << op_name << " without fallback/ms_dispatch";
  return PyboostArgsCallerNoDispatch<OUT_NUM>(func, op_name, args);
}

#define PYBOOST_CALLER_WITH_NAME(OUT_NUM, FUNC, NAME) \
  ([](const py::args &args) -> py::object { return ms::pynative::PyboostArgsCaller<OUT_NUM>(FUNC, NAME, args); })
#define PYBOOST_CALLER(OUT_NUM, FUNC) PYBOOST_CALLER_WITH_NAME(OUT_NUM, FUNC, #FUNC)
}  // namespace pynative
}  // namespace ms
#endif  // MINDSPORE_CCSRC_EXTENSION_PYBOOST_EXTENSION_H_
