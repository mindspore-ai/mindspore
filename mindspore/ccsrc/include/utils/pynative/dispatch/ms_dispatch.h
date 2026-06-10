/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_UTILS_PYNATIVE_DISPATCH_MS_DISPATCH_H_
#define MINDSPORE_CCSRC_INCLUDE_UTILS_PYNATIVE_DISPATCH_MS_DISPATCH_H_

#include <string>
#include <functional>
#include <optional>
#include <vector>
#include <tuple>
#include <utility>
#include "pybind11/pybind11.h"
#include "include/utils/visible.h"
#include "include/utils/pynative/dispatch/dispatch_stack.h"
#include "include/utils/pynative/dispatch/py_convert.h"
#include "include/utils/python_attr.h"
#include "include/utils/tensor_py.h"
#include "ir/tensor.h"
#include "ir/value.h"

namespace py = pybind11;
namespace mindspore {
namespace pynative {

/// \brief A callable wrapper exposed to Python as the 'func' argument in __ms_dispatch__.
/// This class wraps a C++ callable and exposes it to Python with:
/// - `name` property: returns the operator name
/// - `__call__(*args)`: invokes the wrapped callable
/// Python signature: def __ms_dispatch__(cls, func, args=(), kwargs=None)
/// where func is an OpFunc instance.
class COMMON_EXPORT OpFunc {
 public:
  using PyCallable = std::function<py::object(const py::args &)>;

  /// \brief Construct an OpFunc with a callable and operator name.
  /// \param py_callable A callable that accepts py::args and returns py::object.
  /// \param op_name The name of the operator.
  OpFunc(PyCallable py_callable, std::string op_name)
      : py_callable_(std::move(py_callable)), op_name_(std::move(op_name)) {}

  const std::string &name() const { return op_name_; }

  /// \brief Invoke the stored callable from Python with args.
  /// This method is exposed as __call__ in Python.
  /// \param args Positional arguments from Python.
  /// \return The result as a py::object.
  py::object operator()(const py::args &args) const { return py_callable_(args); }

 private:
  PyCallable py_callable_;
  std::string op_name_;
};

COMMON_EXPORT py::object MakeOpFunc(const std::string &name, const py::object &callback);

COMMON_EXPORT bool ms_dispatch_enabled();
COMMON_EXPORT void SetMsDispatchEnabled(bool enabled);

namespace detail {
inline bool CheckMsDispatch(const tensor::TensorPtr &tensor) { return tensor != nullptr && tensor->has_ms_dispatch(); }

inline bool CheckMsDispatch(const std::optional<tensor::TensorPtr> &tensor) {
  return tensor.has_value() && tensor.value() != nullptr && tensor.value()->has_ms_dispatch();
}

inline bool CheckMsDispatch(const std::vector<tensor::TensorPtr> &tensors) {
  for (const auto &t : tensors) {
    if (t != nullptr && t->has_ms_dispatch()) {
      return true;
    }
  }
  return false;
}

inline bool CheckMsDispatch(const ValueTuplePtr &value_tuple) {
  if (value_tuple == nullptr) {
    return false;
  }
  for (const auto &val : value_tuple->value()) {
    if (val != nullptr && val->isa<tensor::Tensor>()) {
      auto tensor = val->cast<tensor::TensorPtr>();
      if (tensor->has_ms_dispatch()) {
        return true;
      }
    }
  }
  return false;
}

inline bool CheckMsDispatch(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>()) {
    return CheckMsDispatch(value->cast<tensor::TensorPtr>());
  } else if (value->isa<ValueTuple>()) {
    return CheckMsDispatch(value->cast<ValueTuplePtr>());
  }
  MS_LOG(DEBUG) << "Not found callback attribute for value: " << value->ToString();
  return false;
}

template <typename T>
inline bool CheckMsDispatch(const T &) {
  return false;
}

COMMON_EXPORT py::object GetMsDispatchFromArgs(const py::args &packed_args);

/// \brief Invoke guard chain using DispatchGuardStack's current_index.
/// This function starts the guard chain. When func(*args) is called in a guard's callback,
/// it gets the next guard from the stack (current_index decremented automatically).
/// When no more guards (current_index < 0), calls the original C++ callback.
/// \param cpp_callback The original C++ callback to call when chain is done
/// \param args The arguments to pass
/// \param op_name The operator name
template <typename CppCallbackWrapper>
py::object InvokeGuardChain(CppCallbackWrapper cpp_callback, const py::args &args, const std::string &op_name) {
  auto &stack = DispatchGuardStack::Get();
  if (stack.NoMoreGuard()) {
    MS_LOG(DEBUG) << "Guard chain complete, executing original op: " << op_name;
    return cpp_callback(args);
  }
  const py::object &current_guard = stack.GetCurrentGuardAndMoveNext();
  // OPT: Use static constexpr for attribute name and FastGetPyObjectAttr
  // instead of py::hasattr + py::getattr for better performance
  static constexpr const char *kCallbackAttr = "__ms_dispatch__";
  py::object handler = FastGetPyObjectAttr(current_guard.ptr(), kCallbackAttr);
  if (handler.ptr() == nullptr) {
    MS_LOG(EXCEPTION) << "Guard does not have valid " << kCallbackAttr << " attribute";
  }

  auto op_func =
    MakeOpFunc(op_name, py::cpp_function([cb = std::move(cpp_callback), op_name](const py::args &inner_args) mutable {
                 return InvokeGuardChain(std::move(cb), inner_args, op_name);
               }));
  MS_LOG(DEBUG) << "Invoke guard: " << op_name;
  return handler(op_func, args);
}

/// \brief Invoke the __ms_dispatch__ handler with OpFunc (for tensor callbacks).
/// This is used for tensor-level callbacks (like DTensor).
/// When func(*args) is called, it directly executes the original C++ callback.
template <typename CppCallbackWrapper>
py::object InvokeMsDispatch(const py::object &handler, CppCallbackWrapper &&cpp_callback, const py::args &packed_args,
                            const std::string &op_name) {
  auto op_func = MakeOpFunc(op_name, py::cpp_function([cb = std::forward<CppCallbackWrapper>(cpp_callback)](
                                                        const py::args &args) mutable { return cb(args); }));
  MS_LOG(DEBUG) << "Invoke tensor-level ms dispatch: " << op_name;
  return handler(op_func, packed_args, py::none());
}
}  // namespace detail

/// \brief Check if any argument has ms_dispatch attribute.
/// This variadic template function checks all arguments to see if any
/// TensorPtr has the ms_dispatch flag set.
/// \param args Variable number of arguments (TensorPtr, ScalarPtr, etc.)
/// \return true if any argument has ms_dispatch; false otherwise.
template <typename... Args>
inline bool HasMsDispatch(const Args &...args) {
  if (!ms_dispatch_enabled()) {
    return false;
  }
  return (detail::CheckMsDispatch(args) || ...);
}

/// \brief Check if dispatch callback should be invoked.
/// This function checks both:
/// 1. Whether there is an active dispatch guard on the stack (highest priority)
/// 2. Whether any tensor argument has __ms_dispatch__ attribute
/// The dispatch guard stack takes priority over tensor-level callbacks.
/// \param args Variable number of arguments (TensorPtr, ScalarPtr, etc.)
/// \return true if dispatch callback should be invoked; false otherwise.
template <typename... Args>
inline bool ShouldDispatch(const Args &...args) {
  if (!ms_dispatch_enabled()) {
    return false;
  }
  // First check if there is an active dispatch guard
  if (HasActiveDispatchGuard()) {
    return true;
  }
  // Then check tensor-level ms_dispatch
  return HasMsDispatch(args...);
}

/// \brief Handle ms_dispatch with a C++ callback function.
/// This function is used when an input tensor has __ms_dispatch__ attribute.
/// It will:
/// 1. Find the __ms_dispatch__ handler from the input tensors
/// 2. Disable autodiff (NoGradGuard scope)
/// 3. Call Python's __ms_dispatch__ handler, passing the C++ callback and packed args
/// 4. The Python handler can invoke the C++ callback to execute the original op
/// 5. After callback returns, autodiff is restored
/// \tparam Callback The callback type (typically a lambda accepting py::args).
/// \param callback A callable that accepts py::args and returns the op result.
/// \param packed_args The packed arguments as py::args.
/// \param op_name The operator name.
/// \return The result from the callback (the return type is deduced from callback).
/// \brief Handle guard chain dispatch.
/// This function processes all guards in the chain, then calls the original callback.
/// Uses GuardChainScope to initialize current_index and prevent re-triggering.
/// \param callback The original C++ callback
/// \param packed_args The packed arguments
/// \param op_name The operator name
/// \return The result from the callback
template <typename Callback>
auto HandleGuardChainDispatch(Callback &&callback, const py::args &packed_args,
                              const std::string &op_name) -> decltype(callback(packed_args)) {
  // RAII: StartChain() sets current_index = stack.size()-1, EndChain() resets to -1
  GuardChainScope guard_chain_scope;

  MS_LOG(DEBUG) << "HandleGuardChainDispatch: processing guards for " << op_name;

  // Create a wrapper that can be called from Python
  auto cpp_callback_wrapper = [&callback](const py::args &args) -> py::object {
    auto result = callback(args);
    return PackValue(result);
  };

  // Process the guard chain (InvokeGuardChain uses current_index internally)
  py::object py_result = detail::InvokeGuardChain(cpp_callback_wrapper, packed_args, op_name);

  // Convert result back to C++ type
  using ReturnType = decltype(callback(packed_args));
  return UnpackValue<ReturnType>(py_result);
}

/// \brief Handle tensor-level ms_dispatch.
/// This function handles tensor-level callbacks (like DTensor).
/// Unlike guard chain dispatch, this doesn't set any special state.
/// \param callback The original C++ callback
/// \param packed_args The packed arguments
/// \param ms_dispatch_handler The tensor's __ms_dispatch__ handler
/// \param op_name The operator name
/// \return The result from the callback
template <typename Callback>
auto HandleTensorFallback(Callback &&callback, const py::args &packed_args, const py::object &ms_dispatch_handler,
                          const std::string &op_name) -> decltype(callback(packed_args)) {
  // Create a wrapper that can be called from Python
  auto cpp_callback_wrapper = [&callback](const py::args &args) -> py::object {
    auto result = callback(args);
    return PackValue(result);
  };

  // Call the tensor's __ms_dispatch__ handler
  py::object py_result = detail::InvokeMsDispatch(ms_dispatch_handler, cpp_callback_wrapper, packed_args, op_name);

  // Convert result back to C++ type
  using ReturnType = decltype(callback(packed_args));
  return UnpackValue<ReturnType>(py_result);
}

/// \brief Handle ms_dispatch dispatch (unified entry point).
/// This function decides whether to use guard chain dispatch or tensor-level dispatch
/// based on the is_guard_callback flag from PackArgs.
/// \param callback The original C++ callback
/// \param packed_args The packed arguments
/// \param dispatch_callback The callback to use
/// \param is_guard_callback Whether this is a guard callback (from PackArgs)
/// \param op_name The operator name
/// \return The result from the callback
template <typename Callback>
auto HandleMsDispatch(Callback &&callback, const py::args &packed_args, const py::object &dispatch_callback,
                      bool is_guard_callback, const std::string &op_name) -> decltype(callback(packed_args)) {
  if (is_guard_callback) {
    // Use guard chain dispatch
    return HandleGuardChainDispatch(std::forward<Callback>(callback), packed_args, op_name);
  } else {
    // Use tensor-level dispatch
    return HandleTensorFallback(std::forward<Callback>(callback), packed_args, dispatch_callback, op_name);
  }
}

inline py::object GetMsDispatchFromObj(PyObject *py_args) {
  if (tensor::IsPyObjectTensorPy(py_args)) {
    auto t = tensor::ConvertPyObjectToTensor(py_args);
    return GetMsDispatch(t);
  } else if (PyTuple_Check(py_args)) {
    Py_ssize_t size = PyTuple_Size(py_args);
    for (Py_ssize_t i = 0; i < size; ++i) {
      PyObject *item = PyTuple_GetItem(py_args, i);
      auto ms_dispatch = GetMsDispatchFromObj(item);
      if (!ms_dispatch.is_none()) {
        MS_LOG(DEBUG) << "Get ms_dispatch from " << Py_TYPE(item)->tp_name;
        return ms_dispatch;
      }
    }
    return py::none();
  } else if (PyList_Check(py_args)) {
    Py_ssize_t size = PyList_Size(py_args);
    for (Py_ssize_t i = 0; i < size; ++i) {
      PyObject *item = PyList_GetItem(py_args, i);
      auto ms_dispatch = GetMsDispatchFromObj(item);
      if (!ms_dispatch.is_none()) {
        MS_LOG(DEBUG) << "Get ms_dispatch from " << Py_TYPE(item)->tp_name;
        return ms_dispatch;
      }
    }
    return py::none();
  } else {
    MS_LOG(DEBUG) << "Cannot get __ms_dispatch__ attr from: " << Py_TYPE(py_args)->tp_name
                  << ". Supported type is <Tensor, tuple, list>";
    return py::none();
  }
}

inline py::object GetMsDispatchFromInput(const py::args &args) { return GetMsDispatchFromObj(args.ptr()); }

/// \brief Get the dispatch callback from py::args with dispatch guard priority.
/// This function checks dispatch priority:
/// 1. Dispatch guard stack (if active and not in chain)
/// 2. Tensor-level __ms_dispatch__ attribute
/// \param args The py::args to search for callback.
/// \return A tuple of (callback, is_guard_callback).
inline std::tuple<py::object, bool> GetDispatchCallbackFromInput(const py::args &args) {
  // First check dispatch guard stack
  if (HasActiveDispatchGuard()) {
    // Return empty callback - guard chain will be handled separately
    return std::make_tuple(py::none(), true);
  }
  // Fall back to tensor-level callback
  return std::make_tuple(GetMsDispatchFromInput(args), false);
}

}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_UTILS_PYNATIVE_DISPATCH_MS_DISPATCH_H_
