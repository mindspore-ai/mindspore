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

#ifndef MINDSPORE_CCSRC_INCLUDE_UTILS_PYNATIVE_DISPATCH_PY_CONVERT_H_
#define MINDSPORE_CCSRC_INCLUDE_UTILS_PYNATIVE_DISPATCH_PY_CONVERT_H_

#include <string>
#include <optional>
#include <vector>
#include <tuple>
#include <utility>
#include "pybind11/pybind11.h"
#include "include/utils/pynative/dispatch/dispatch_stack.h"
#include "include/utils/convert_utils_py.h"
#include "include/utils/tensor_py.h"
#include "ir/tensor.h"
#include "ir/scalar.h"
#include "ir/value.h"

namespace py = pybind11;
namespace mindspore {
namespace pynative {

inline const std::string &GetMsDispatchStr() {
  static const std::string kMsDispatch = "__ms_dispatch__";
  return kMsDispatch;
}

inline const std::string &GetSourcePyObjStr() {
  static const std::string kSourcePyObj = "source_py_obj";
  return kSourcePyObj;
}

inline py::object PackValue(const tensor::TensorPtr &tensor) {
  if (tensor == nullptr) {
    return py::none();
  }
  if (tensor->has_ms_dispatch()) {
    auto py_data = tensor->user_data<tensor::TensorPyUserData>(GetSourcePyObjStr());
    MS_EXCEPTION_IF_NULL(py_data);
    return py_data->obj;
  }
  return tensor::PackTensorToPyObject(tensor);
}

inline py::object PackValue(const ScalarPtr &scalar) {
  if (scalar == nullptr) {
    return py::none();
  }
  return ScalarPtrToPyData(scalar);
}

inline py::object PackValue(const std::optional<tensor::TensorPtr> &opt_tensor) {
  if (!opt_tensor.has_value() || opt_tensor.value() == nullptr) {
    return py::none();
  }
  return tensor::PackTensorToPyObject(opt_tensor.value());
}

inline py::object PackValue(const std::vector<tensor::TensorPtr> &tensors) {
  py::list result;
  for (const auto &t : tensors) {
    result.append(PackValue(t));
  }
  return result;
}

namespace detail {
template <typename... Ts, size_t... Is>
inline py::tuple PackTupleImpl(const std::tuple<Ts...> &tuple_val, std::index_sequence<Is...>) {
  return py::make_tuple(PackValue(std::get<Is>(tuple_val))...);
}
}  // namespace detail

#define TP tensor::TensorPtr
#define DEFINE_PACK_TENSOR_TUPLE(...)                                                                                \
  inline py::object PackValue(const std::tuple<__VA_ARGS__> &tuple_val) {                                            \
    return detail::PackTupleImpl(tuple_val, std::make_index_sequence<std::tuple_size_v<std::tuple<__VA_ARGS__>>>{}); \
  }

DEFINE_PACK_TENSOR_TUPLE(TP, TP)
DEFINE_PACK_TENSOR_TUPLE(TP, TP, TP)
DEFINE_PACK_TENSOR_TUPLE(TP, TP, TP, TP)
DEFINE_PACK_TENSOR_TUPLE(TP, TP, TP, TP, TP)
DEFINE_PACK_TENSOR_TUPLE(TP, TP, TP, TP, TP, TP)
DEFINE_PACK_TENSOR_TUPLE(TP, TP, TP, TP, TP, TP, TP)
#undef DEFINE_PACK_TENSOR_TUPLE
#undef TP

inline py::object PackValue(const Int64ImmPtr &imm) {
  if (imm == nullptr) {
    return py::none();
  }
  return py::cast(imm->value());
}

inline py::object PackValue(const std::optional<Int64ImmPtr> &opt_imm) {
  if (!opt_imm.has_value() || opt_imm.value() == nullptr) {
    return py::none();
  }
  return py::cast(opt_imm.value()->value());
}

inline py::object PackValue(const BoolImmPtr &imm) {
  if (imm == nullptr) {
    return py::none();
  }
  return py::cast(imm->value());
}

inline py::object PackValue(const FP32ImmPtr &imm) {
  if (imm == nullptr) {
    return py::none();
  }
  return py::cast(imm->value());
}

inline py::object PackValue(const std::optional<FP32ImmPtr> &opt_imm) {
  if (!opt_imm.has_value() || opt_imm.value() == nullptr) {
    return py::none();
  }
  return py::cast(opt_imm.value()->value());
}

inline py::object PackValue(const StringImmPtr &imm) {
  if (imm == nullptr) {
    return py::none();
  }
  return py::cast(imm->value());
}

inline py::object PackValue(const std::optional<StringImmPtr> &opt_imm) {
  if (!opt_imm.has_value() || opt_imm.value() == nullptr) {
    return py::none();
  }
  return py::cast(opt_imm.value()->value());
}

inline py::object PackValue(const std::optional<ScalarPtr> &opt_scalar) {
  if (!opt_scalar.has_value() || opt_scalar.value() == nullptr) {
    return py::none();
  }
  return ScalarPtrToPyData(opt_scalar.value());
}

inline py::object PackValue(const ValueTuplePtr &value_tuple);

inline py::object PackValue(const ValuePtr &value) {
  if (value == nullptr || value->isa<None>()) {
    return py::none();
  }
  if (value->isa<tensor::Tensor>()) {
    return PackValue(value->cast<tensor::TensorPtr>());
  }
  if (value->isa<Scalar>()) {
    return PackValue(value->cast<ScalarPtr>());
  }
  if (value->isa<StringImm>()) {
    return PackValue(value->cast<StringImmPtr>());
  }
  if (value->isa<ValueTuple>()) {
    auto value_tuple = value->cast<ValueTuplePtr>();
    return PackValue(value_tuple);
  }

  MS_LOG(EXCEPTION) << "PackValue not supported for type: " << value->ToString();
}

inline py::object PackValue(const ValueTuplePtr &value_tuple) {
  MS_EXCEPTION_IF_NULL(value_tuple);
  const auto &values = value_tuple->value();
  py::list result;
  for (const auto &val : values) {
    MS_EXCEPTION_IF_NULL(val);
    result.append(PackValue(val));
  }
  return py::tuple(result);
}

inline py::object PackValue(const std::optional<ValueTuplePtr> &opt_tuple) {
  if (!opt_tuple.has_value() || opt_tuple.value() == nullptr) {
    return py::none();
  }
  return PackValue(opt_tuple.value());
}

inline py::object PackValue(int64_t value) { return py::cast(value); }

inline py::object PackValue(const std::vector<int64_t> &values) {
  py::list result;
  for (const auto &v : values) {
    result.append(py::cast(v));
  }
  return result;
}

inline py::object GetMsDispatch(const tensor::TensorPtr &tensor) {
  if (tensor == nullptr || !tensor->has_ms_dispatch()) {
    return py::none();
  }
  const auto user_data = tensor->user_data<tensor::TensorPyUserData>(GetMsDispatchStr());
  if (user_data == nullptr) {
    MS_LOG(EXCEPTION) << "Tensor " << tensor->ToString() << " should have dispatch attribute";
  }
  return user_data->obj;
}

/// \brief Get the original PyObject from tensor's user_data.
/// This retrieves the original Python Tensor subclass object that was stored
/// when the tensor was converted with ms_dispatch support.
/// \param tensor The tensor to get the original PyObject from.
/// \return The original PyObject, or py::none() if not found.
inline py::object GetSourcePyObj(const tensor::TensorPtr &tensor) {
  if (tensor == nullptr) {
    return py::none();
  }
  const auto user_data = tensor->user_data<tensor::TensorPyUserData>(GetSourcePyObjStr());
  MS_EXCEPTION_IF_NULL(user_data);
  return user_data->obj;
}

/// \brief Create a shallow copy of tensor with source_py_obj stored in user_data.
/// This is used when converting a Python Tensor subclass with __ms_dispatch__ attribute
/// to preserve the original PyObject reference for later dispatch.
/// \param original_tensor The tensor to wrap.
/// \param py_obj The original Python object to store.
/// \return A new tensor with the original PyObject stored in user_data.
inline tensor::TensorPtr WrapTensorWithSourcePyObj(const tensor::TensorPtr &original_tensor, const py::object &py_obj) {
  if (original_tensor == nullptr) {
    return nullptr;
  }

  auto new_tensor = std::make_shared<tensor::Tensor>(*original_tensor);
  if (original_tensor->is_parameter()) {
    new_tensor->set_param_info(original_tensor->param_info());
  }

  auto py_obj_data = std::make_shared<tensor::TensorPyUserData>();
  py_obj_data->obj = py_obj;
  new_tensor->set_user_data<tensor::TensorPyUserData>(GetSourcePyObjStr(), py_obj_data);

  return new_tensor;
}

template <typename T>
inline py::object GetMsDispatch(const T &) {
  return py::none();
}

inline py::object GetMsDispatch(const std::optional<tensor::TensorPtr> &opt_tensor) {
  if (!opt_tensor.has_value()) {
    return py::none();
  }
  return GetMsDispatch(opt_tensor.value());
}

inline py::object GetMsDispatch(const std::vector<tensor::TensorPtr> &tensors) {
  for (const auto &t : tensors) {
    auto callback = GetMsDispatch(t);
    if (!callback.is_none()) {
      return callback;
    }
  }
  return py::none();
}

inline py::object GetMsDispatch(const ValueTuplePtr &value_tuple) {
  if (value_tuple == nullptr) {
    return py::none();
  }
  const auto &values = value_tuple->value();
  for (const auto &val : values) {
    MS_EXCEPTION_IF_NULL(val);
    if (val->isa<tensor::Tensor>()) {
      auto tensor = val->cast<tensor::TensorPtr>();
      auto callback = GetMsDispatch(tensor);
      if (!callback.is_none()) {
        return callback;
      }
    }
  }
  return py::none();
}

namespace detail {
inline py::object FindFirstCallback() { return py::none(); }

template <typename T, typename... Rest>
inline py::object FindFirstCallback(const T &first, const Rest &...rest) {
  auto callback = GetMsDispatch(first);
  if (!callback.is_none()) {
    return callback;
  }
  return FindFirstCallback(rest...);
}
}  // namespace detail

/// \brief Pack multiple C++ values to py::args and check dispatch type.
/// This function checks dispatch priority:
/// 1. Dispatch guard stack (if active and not in chain)
/// 2. Tensor-level __ms_dispatch__ attribute
/// \return A tuple of (packed_args, dispatch_callback, is_guard_callback).
template <typename... Args>
inline std::tuple<py::args, py::object, bool> PackArgs(const Args &...args) {
  py::tuple t = py::make_tuple(PackValue(args)...);
  if (HasActiveDispatchGuard()) {
    MS_LOG(DEBUG) << "PackArgs: using guard chain dispatch";
    // Return empty callback - HandleGuardChainDispatch will handle it
    return std::make_tuple(py::args(t), py::none(), true);
  }

  // Fall back to tensor-level callback
  py::object callback = detail::FindFirstCallback(args...);
  if (callback.is_none()) {
    MS_LOG(EXCEPTION) << "No __ms_dispatch__ found for the given arguments";
  }
  MS_LOG(DEBUG) << "PackArgs: using tensor-level callback";
  return std::make_tuple(py::args(t), callback, false);
}

// Forward declaration - generic template is deleted to ensure all types are explicitly handled
template <typename T>
T UnpackValue(const py::object &obj) = delete;

template <>
inline tensor::TensorPtr UnpackValue<tensor::TensorPtr>(const py::object &obj) {
  if (obj.is_none()) {
    return nullptr;
  }

  auto original_tensor = tensor::ConvertPyObjectToTensor(obj.ptr());
  return WrapTensorWithSourcePyObj(original_tensor, obj);
}

template <>
inline std::optional<tensor::TensorPtr> UnpackValue<std::optional<tensor::TensorPtr>>(const py::object &obj) {
  if (obj.is_none()) {
    return std::nullopt;
  }
  return UnpackValue<tensor::TensorPtr>(obj);
}

template <>
inline std::vector<tensor::TensorPtr> UnpackValue<std::vector<tensor::TensorPtr>>(const py::object &obj) {
  std::vector<tensor::TensorPtr> result;
  if (obj.is_none()) {
    return result;
  }
  py::list py_list = py::cast<py::list>(obj);
  for (const auto &item : py_list) {
    result.push_back(UnpackValue<tensor::TensorPtr>(py::cast<py::object>(item)));
  }
  return result;
}

template <>
inline ScalarPtr UnpackValue<ScalarPtr>(const py::object &obj) {
  if (obj.is_none()) {
    return nullptr;
  }
  // ScalarPtrToPyData returns Python native types (int, float, bool)
  // We need to convert them back to Scalar
  if (py::isinstance<py::bool_>(obj)) {
    return std::make_shared<BoolImm>(obj.cast<bool>());
  } else if (py::isinstance<py::int_>(obj)) {
    return std::make_shared<Int64Imm>(obj.cast<int64_t>());
  } else if (py::isinstance<py::float_>(obj)) {
    return std::make_shared<FP32Imm>(obj.cast<float>());
  }
  return nullptr;
}

template <>
inline std::optional<ScalarPtr> UnpackValue<std::optional<ScalarPtr>>(const py::object &obj) {
  if (obj.is_none()) {
    return std::nullopt;
  }
  // ScalarPtrToPyData returns Python native types (int, float, bool)
  // We need to convert them back to Scalar
  if (py::isinstance<py::bool_>(obj)) {
    return std::make_shared<BoolImm>(obj.cast<bool>());
  } else if (py::isinstance<py::int_>(obj)) {
    return std::make_shared<Int64Imm>(obj.cast<int64_t>());
  } else if (py::isinstance<py::float_>(obj)) {
    return std::make_shared<FP32Imm>(obj.cast<float>());
  }
  return std::nullopt;
}

template <>
inline Int64ImmPtr UnpackValue<Int64ImmPtr>(const py::object &obj) {
  if (obj.is_none()) {
    return nullptr;
  }
  auto value = obj.cast<int64_t>();
  return std::make_shared<Int64Imm>(value);
}

template <>
inline std::optional<Int64ImmPtr> UnpackValue<std::optional<Int64ImmPtr>>(const py::object &obj) {
  if (obj.is_none()) {
    return std::nullopt;
  }
  auto value = obj.cast<int64_t>();
  return std::make_shared<Int64Imm>(value);
}

template <>
inline BoolImmPtr UnpackValue<BoolImmPtr>(const py::object &obj) {
  if (obj.is_none()) {
    return nullptr;
  }
  auto value = obj.cast<bool>();
  return std::make_shared<BoolImm>(value);
}

template <>
inline FP32ImmPtr UnpackValue<FP32ImmPtr>(const py::object &obj) {
  if (obj.is_none()) {
    return nullptr;
  }
  auto value = obj.cast<float>();
  return std::make_shared<FP32Imm>(value);
}

template <>
inline std::optional<FP32ImmPtr> UnpackValue<std::optional<FP32ImmPtr>>(const py::object &obj) {
  if (obj.is_none()) {
    return std::nullopt;
  }
  auto value = obj.cast<float>();
  return std::make_shared<FP32Imm>(value);
}

template <>
inline StringImmPtr UnpackValue<StringImmPtr>(const py::object &obj) {
  if (obj.is_none()) {
    return nullptr;
  }
  auto value = obj.cast<std::string>();
  return std::make_shared<StringImm>(value);
}

template <>
inline std::optional<StringImmPtr> UnpackValue<std::optional<StringImmPtr>>(const py::object &obj) {
  if (obj.is_none()) {
    return std::nullopt;
  }
  auto value = obj.cast<std::string>();
  return std::make_shared<StringImm>(value);
}

namespace detail {
// Helper function to convert Python object to ValuePtr for ValueTuple elements
inline ValuePtr PyObjectToValueForTuple(const py::object &obj) {
  if (obj.is_none()) {
    return kNone;
  }
  if (tensor::IsTensorPy(obj)) {
    // Check if tensor has __ms_dispatch__ attribute (like DTensor subclass)
    auto tensor = tensor::ConvertToTensor(obj);
    if (tensor != nullptr && tensor->has_ms_dispatch()) {
      // Wrap with source_py_obj to preserve the original PyObject reference
      return WrapTensorWithSourcePyObj(tensor, obj);
    }
    return tensor;
  }
  // Note: check bool before int because Python bool is subclass of int
  if (py::isinstance<py::bool_>(obj)) {
    return std::make_shared<BoolImm>(obj.cast<bool>());
  }
  if (py::isinstance<py::int_>(obj)) {
    return std::make_shared<Int64Imm>(obj.cast<int64_t>());
  }
  if (py::isinstance<py::float_>(obj)) {
    return std::make_shared<FP32Imm>(obj.cast<float>());
  }
  if (py::isinstance<py::str>(obj)) {
    return std::make_shared<StringImm>(obj.cast<std::string>());
  }
  if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
    std::vector<ValuePtr> elements;
    for (const auto &item : obj) {
      elements.push_back(PyObjectToValueForTuple(py::reinterpret_borrow<py::object>(item)));
    }
    return std::make_shared<ValueTuple>(elements);
  }
  return nullptr;
}
}  // namespace detail

template <>
inline ValueTuplePtr UnpackValue<ValueTuplePtr>(const py::object &obj) {
  if (obj.is_none()) {
    return nullptr;
  }
  if (!py::isinstance<py::tuple>(obj) && !py::isinstance<py::list>(obj)) {
    return nullptr;
  }
  std::vector<ValuePtr> elements;
  for (const auto &item : obj) {
    elements.push_back(detail::PyObjectToValueForTuple(py::reinterpret_borrow<py::object>(item)));
  }
  return std::make_shared<ValueTuple>(elements);
}

template <>
inline std::optional<ValueTuplePtr> UnpackValue<std::optional<ValueTuplePtr>>(const py::object &obj) {
  if (obj.is_none()) {
    return std::nullopt;
  }
  if (!py::isinstance<py::tuple>(obj) && !py::isinstance<py::list>(obj)) {
    return std::nullopt;
  }
  std::vector<ValuePtr> elements;
  for (const auto &item : obj) {
    elements.push_back(detail::PyObjectToValueForTuple(py::reinterpret_borrow<py::object>(item)));
  }
  return std::make_shared<ValueTuple>(elements);
}

template <>
inline int64_t UnpackValue<int64_t>(const py::object &obj) {
  return py::cast<int64_t>(obj);
}

template <>
inline std::vector<int64_t> UnpackValue<std::vector<int64_t>>(const py::object &obj) {
  if (obj.is_none()) {
    return {};
  }
  std::vector<int64_t> result;
  py::list py_list = py::cast<py::list>(obj);
  for (const auto &item : py_list) {
    result.push_back(py::cast<int64_t>(item));
  }
  return result;
}

namespace detail {
template <size_t N>
struct TensorTuple;

template <size_t N, size_t... Is>
inline auto UnpackTensorTupleImpl(const py::tuple &py_tuple, std::index_sequence<Is...>) {
  return typename TensorTuple<N>::type{UnpackValue<tensor::TensorPtr>(py::cast<py::object>(py_tuple[Is]))...};
}
}  // namespace detail

#define TP tensor::TensorPtr
#define DEFINE_UNPACK_TENSOR_TUPLE(N, ...)                                                     \
  namespace detail {                                                                           \
  template <>                                                                                  \
  struct TensorTuple<N> {                                                                      \
    using type = std::tuple<__VA_ARGS__>;                                                      \
  };                                                                                           \
  }                                                                                            \
  template <>                                                                                  \
  inline std::tuple<__VA_ARGS__> UnpackValue<std::tuple<__VA_ARGS__>>(const py::object &obj) { \
    py::tuple py_tuple = py::cast<py::tuple>(obj);                                             \
    return detail::UnpackTensorTupleImpl<N>(py_tuple, std::make_index_sequence<N>{});          \
  }

DEFINE_UNPACK_TENSOR_TUPLE(2, TP, TP)
DEFINE_UNPACK_TENSOR_TUPLE(3, TP, TP, TP)
DEFINE_UNPACK_TENSOR_TUPLE(4, TP, TP, TP, TP)
DEFINE_UNPACK_TENSOR_TUPLE(5, TP, TP, TP, TP, TP)
DEFINE_UNPACK_TENSOR_TUPLE(6, TP, TP, TP, TP, TP, TP)
DEFINE_UNPACK_TENSOR_TUPLE(7, TP, TP, TP, TP, TP, TP, TP)
#undef DEFINE_UNPACK_TENSOR_TUPLE
#undef TP

/// \brief Unpack a py::object at specific index from py::args.
template <typename T>
inline T UnpackArg(const py::args &args, size_t index) {
  return UnpackValue<T>(py::cast<py::object>(args[index]));
}
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_UTILS_PYNATIVE_DISPATCH_PY_CONVERT_H_
