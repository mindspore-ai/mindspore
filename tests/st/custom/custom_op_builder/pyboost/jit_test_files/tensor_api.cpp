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

#include "ms_extension/api.h"

using PyboostRunner = ms::pynative::PyboostRunner;
using TypeId = ms::TypeId;

ms::Tensor reshape_fp32(const ms::Tensor &x, const ShapeVector &shape) {
  if (x.is_contiguous()) {
    throw std::invalid_argument("For reshape_fp32, the inputs should be uncontiguous tensor.");
  }
  return x.contiguous().reshape(shape).cast(ms::TypeId::kNumberTypeFloat32);
}

auto pyboost_reshape(const ms::Tensor &x, const ShapeVector &shape) {
  x.SetNeedContiguous(false);
  return PyboostRunner::Call<1>(reshape_fp32, x, shape);
}

ms::TypeId type_str_to_type_id(const std::string &dtype) {
  std::map<std::string, TypeId> m = {{"int32", TypeId::kNumberTypeInt32},
                                     {"int64", TypeId::kNumberTypeInt64},
                                     {"float16", TypeId::kNumberTypeFloat16},
                                     {"float32", TypeId::kNumberTypeFloat32}};
  auto iter = m.find(dtype);
  if (iter == m.end()) {
    throw std::invalid_argument(dtype + " is not supported.");
  }
  return iter->second;
}

template <typename T>
ms::Tensor tensor_value(T v, const std::string &dtype) {
  return ms::tensor(v, type_str_to_type_id(dtype)).contiguous();
}
template <typename T>
py::object pyboost_tensor_value(T v, const std::string &dtype) {
  return PyboostRunner::Call<1>(tensor_value<T>, v, dtype);
}

ms::Tensor ones(const ShapeVector &shape, const std::string &dtype) {
  return ms::ones(shape, type_str_to_type_id(dtype));
}
auto pyboost_ones(const ShapeVector &shape, const std::string &dtype) {
  return ms::pynative::PyboostRunner::Call<1>(ones, shape, dtype);
}

ms::Tensor zeros(const ShapeVector &shape, const std::string &dtype) {
  return ms::zeros(shape, type_str_to_type_id(dtype));
}
auto pyboost_zeros(const ShapeVector &shape, const std::string &dtype) {
  return ms::pynative::PyboostRunner::Call<1>(zeros, shape, dtype);
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("reshape_fp32", &pyboost_reshape, "reshape and cast to fp32");
  m.def("tensor_int", &pyboost_tensor_value<int64_t>, "tensor_int");
  m.def("tensor_double", &pyboost_tensor_value<double>, "tensor_double");
  m.def("tensor_int_list", &pyboost_tensor_value<std::vector<int64_t>>, "tensor_int_list");
  m.def("tensor_double_list", &pyboost_tensor_value<std::vector<double>>, "tensor_double_list");
  m.def("ones", &pyboost_ones, "ones");
  m.def("zeros", &pyboost_zeros, "zeros");
}
