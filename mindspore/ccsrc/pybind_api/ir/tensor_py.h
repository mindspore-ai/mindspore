/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_UTILS_TENSOR_PY_H_
#define MINDSPORE_CCSRC_UTILS_TENSOR_PY_H_

#include <memory>
#include <string>
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include "ir/tensor.h"

namespace py = pybind11;

namespace pybind11 {
namespace detail {
// Similar to enums in `pybind11/numpy.h`. Determined by doing:
// python3 -c 'import numpy as np; print(np.dtype(np.float16).num)'
constexpr int NPY_FLOAT16 = 23;

template <typename T>
struct npy_scalar_caster {
  PYBIND11_TYPE_CASTER(T, _("PleaseOverride"));
  using Array = array_t<T>;

  bool load(handle src, bool convert) {
    // Taken from Eigen casters. Permits either scalar dtype or scalar array.
    handle type = dtype::of<T>().attr("type");
    if (!convert && !isinstance<Array>(src) && !isinstance(src, type)) return false;

    Array tmp = Array::ensure(src);
    if (tmp && tmp.size() == 1 && tmp.ndim() == 0) {
      this->value = *tmp.data();
      return true;
    }

    return false;
  }

  static handle cast(T src, return_value_policy, handle) {
    Array tmp({1});
    tmp.mutable_at(0) = src;
    tmp.resize({});

    // You could also just return the array if you want a scalar array.
    object scalar = tmp[tuple()];
    return scalar.release();
  }
};

template <>
struct npy_format_descriptor<float16> {
  static constexpr auto name = "float16";
  static pybind11::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
    return reinterpret_borrow<pybind11::dtype>(ptr);
  }
  virtual ~npy_format_descriptor<float16>() {}
};

template <>
struct type_caster<float16> : public npy_scalar_caster<float16> {
  static constexpr auto name = "float16";
};
}  // namespace detail
}  // namespace pybind11

// brief mindspore namespace.
//
// mindspore namespace is the top level namespace of Mindsporeession project.
// Other namespace should be a sub namespace of mindspore namespace in the ME project.
namespace mindspore {
// brief mindspore::tensor namespace
//
// A sub namespace in ME to support tensor related definition.
namespace tensor {
// Tensor python wrapper and adapter class.
class TensorPy {
 public:
  // brief Create Tensor from a numpy array object.
  //
  // param input [py::array] Data value of the tensor.
  // param data_type [TypeId] Data type of the tensor.
  static TensorPtr MakeTensor(const py::array &input, const TypePtr &data_type = nullptr);

  // brief Create Tensor from a numpy array without copy.
  //
  // param input [py::array] Data value of the tensor.
  static TensorPtr MakeTensorNoCopy(const py::array &input);

  static py::array SyncAsNumpy(const Tensor &tensor);

  static py::array AsNumpy(const Tensor &tensor);

  static py::tuple GetPyTupleShape(const Tensor &tensor);

  static py::tuple GetPyTupleStrides(const Tensor &tensor);

  static py::int_ GetPyItemSize(const Tensor &tensor);

  static py::int_ GetPyNBytes(const Tensor &tensor);

  static void FlushFromCache(const Tensor &tensor);
};
}  // namespace tensor
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_TENSOR_PY_H_
