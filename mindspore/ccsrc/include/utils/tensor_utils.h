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

#ifndef MINDSPORE_MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_TENSOR_UTILS_H_
#define MINDSPORE_MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_TENSOR_UTILS_H_

#include <algorithm>
#include <iterator>
#include <tuple>
#include <vector>
#include "ir/tensor.h"
#include "include/utils/visible.h"
#include "include/utils/tensor_py.h"

namespace mindspore {
namespace tensor {
class COMMON_EXPORT TensorWrapper {
 public:
  explicit TensorWrapper(bool has_side_effect) { empty_tensor_ = PackTensor(nullptr, has_side_effect); }

  PyObject *value() const { return empty_tensor_; }

  const stub::StubNodePtr &MakeFuture() const {
    PyType<TensorPy> *tensor = (PyType<TensorPy> *)empty_tensor_;
    return tensor->value.MakeStub();
  }

 private:
  PyObject *empty_tensor_;
};

// Helper function to create a std::tuple with N elements of type T
template <typename T, size_t N, bool Flag, size_t... I>
auto MakeTupleImpl(std::index_sequence<I...>) {
  // Use fold expression to create a tuple with N default-constructed elements
  return std::make_tuple((void(I), T(Flag))...);
}

// Main function to create a std::tuple with N elements of type T
template <typename T, size_t N, bool Flag>
auto MakeTuple() {
  // Generate an index sequence and delegate to the implementation
  return MakeTupleImpl<T, N, Flag>(std::make_index_sequence<N>{});
}

template <bool Flag>
std::vector<TensorWrapper> MakeVector(size_t num) {
  std::vector<TensorWrapper> py_output;
  py_output.reserve(num);
  for (size_t i = 0; i < num; ++i) {
    py_output.emplace_back(Flag);
  }
  return py_output;
}

// Helper function to apply a transformation to each element of the tuple
template <typename Tuple, size_t... Indices>
void TransformTupleImpl(PyObject *py_tuple, const Tuple &tuple, std::index_sequence<Indices...>) {
  // Create a new tuple by calling `value()` on each element
  (PyTuple_SET_ITEM(py_tuple, Indices, std::get<Indices>(tuple).value()), ...);
}

// Main function to transform a tuple
template <typename Tuple>
auto TransformOutput(const Tuple &tuple) {
  constexpr size_t tuple_size = std::tuple_size_v<Tuple>;

  if constexpr (tuple_size == 1) {
    // If the tuple has only 1 element, return item->value()
    return std::get<0>(tuple).value();
  } else {
    PyObject *py_tuple = PyTuple_New(tuple_size);
    // If the tuple has more than 1 element, return a new tuple
    TransformTupleImpl(py_tuple, tuple, std::make_index_sequence<tuple_size>{});
    return py_tuple;
  }
}

COMMON_EXPORT PyObject *TransformVectorOutput(const std::vector<TensorWrapper> &py_output);

template <typename Tuple>
PyObject *TransformOutput(const Tuple &tuple, PyObject *comm_handle) {
  constexpr size_t tuple_size = std::tuple_size_v<Tuple>;
  PyObject *py_tuple = PyTuple_New(tuple_size + 1);
  TransformTupleImpl(py_tuple, tuple, std::make_index_sequence<tuple_size>{});
  PyTuple_SET_ITEM(py_tuple, tuple_size, comm_handle);
  return py_tuple;
}

template <typename Tuple, size_t... Indices>
auto TransformPromiseImpl(const Tuple &tuple, std::index_sequence<Indices...>) {
  // Create a new tuple by calling `value()` on each element
  return std::make_tuple(std::get<Indices>(tuple).MakeFuture()...);
}

template <typename Tuple>
auto TransformPromise(const Tuple &tuple) {
  constexpr size_t tuple_size = std::tuple_size_v<Tuple>;
  return TransformPromiseImpl(tuple, std::make_index_sequence<tuple_size>{});
}

COMMON_EXPORT std::vector<stub::StubNodePtr> TransformVectorPromise(const std::vector<TensorWrapper> &py_output);

COMMON_EXPORT void SetPromise(const std::tuple<stub::StubNodePtr> &promises, const TensorPtr &tensor);

// Helper function to set values using a compile-time loop
template <typename... T1, typename... T2, std::size_t... I>
void SetPromiseImpl(const std::tuple<T1...> &t1, const std::tuple<T2...> &t2, std::index_sequence<I...>) {
  // Fold expression to set each element of t1 with the corresponding element of t2
  ((std::get<I>(t1)->SetValue(std::get<I>(t2))), ...);
}

// Main function to set values
template <typename... T1, typename... T2>
void SetPromise(const std::tuple<T1...> &t1, const std::tuple<T2...> &t2) {
  // Ensure the tuples have the same size
  static_assert(sizeof...(T1) == sizeof...(T2), "Tuples must have the same size");

  // Call the implementation with an index sequence
  SetPromiseImpl(t1, t2, std::index_sequence_for<T1...>{});
}

COMMON_EXPORT void SetPromise(const std::vector<stub::StubNodePtr> &t1, const std::vector<TensorPtr> &t2);

COMMON_EXPORT void FlattenOutputs(const ValuePtr &value, std::vector<TensorPtr> *outputs);

template <typename... Ts>
std::vector<std::common_type_t<Ts...>> tuple_to_vector(const std::tuple<Ts...> &t) {
  std::vector<std::common_type_t<Ts...>> vec;
  vec.reserve(sizeof...(Ts));

  std::apply([&vec](const auto &...args) { (vec.push_back(args), ...); }, t);

  return vec;
}

template <typename... T>
void SetPromise(const std::tuple<T...> &tuple, const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  std::vector<TensorPtr> outputs;
  FlattenOutputs(value, &outputs);
  std::vector<stub::StubNodePtr> promises = tuple_to_vector(tuple);
  using TupleType = typename std::decay<decltype(tuple)>::type;
  constexpr std::size_t size = std::tuple_size<TupleType>::value;
  if (outputs.size() != size) {
    MS_LOG(EXCEPTION) << "Promise tuple size is " << size << " but outputs size is " << outputs.size();
  }
  for (size_t i = 0; i < size; ++i) {
    promises[i]->SetValue(outputs[i]);
  }
}

template <typename... T1, std::size_t... I>
void SetExceptionImpl(const std::tuple<T1...> &t1, std::index_sequence<I...>) {
  ((std::get<I>(t1)->SetException(std::current_exception())), ...);
}

template <typename... T1>
void SetException(const std::tuple<T1...> &t1) {
  SetExceptionImpl(t1, std::index_sequence_for<T1...>{});
}

COMMON_EXPORT void SetException(const std::vector<stub::StubNodePtr> &t1);

template <typename T>
inline T GetTensorData(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  auto cpu_tensor = tensor->cpu();
  return *(static_cast<T *>(cpu_tensor->data_c()));
}
}  // namespace tensor
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_TENSOR_UTILS_H_
