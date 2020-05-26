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

#ifndef MINDSPORE_CCSRC_IR_TENSOR_H_
#define MINDSPORE_CCSRC_IR_TENSOR_H_

#include <memory>
#include <string>
#include <vector>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

#include "Eigen/Core"
#include "device/device_address.h"
#include "ir/meta_tensor.h"
#include "utils/log_adapter.h"

namespace py = pybind11;

using float16 = Eigen::half;

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

using mindspore::device::DeviceAddress;
using DeviceAddressPtr = std::shared_ptr<mindspore::device::DeviceAddress>;
// brief mindspore namespace.
//
// mindspore namespace is the top level namespace of Mindsporeession project.
// Other namespace should be a sub namespace of mindspore namespace in the ME project.
namespace mindspore {

// brief mindspore::tensor namespace
//
// A sub namespace in ME to support tensor related definition.
namespace tensor {
// Tensor entity class
class Tensor : public MetaTensor {
 public:
  Tensor() = default;
  abstract::AbstractBasePtr ToAbstract() override;
  // brief Constructor for Python.
  //
  // param type_ptr [TypePty] Data type of the tensor.
  // param py_shape [py::tuple] The shape represented by py::tuple of the tensor.
  Tensor(const TypePtr &type_ptr, const py::tuple &shape);

  // brief Constructor for C++.
  //
  // param data_type [TypeId] Data type of the tensor.
  // param shape The shape represented by std::vector<int> of the tensor.
  Tensor(TypeId data_type, const std::vector<int> &shape);

  // brief Constructor for Python.
  //
  // param input [py::array] Data value of the tensor.
  // param data_type [TypeId] Data type of the tensor.
  explicit Tensor(const py::array &input, const TypePtr &data_type = nullptr);

  // brief Constructor
  //
  // param input [py::list] the data for tensor
  // param data_type [TypeId] data type
  explicit Tensor(const py::list &input, const TypePtr &data_type = nullptr);

  // brief Constructor
  //
  // param input [py::tuple] the data for tensor
  // param data_type [TypeId] data type
  explicit Tensor(const py::tuple &input, const TypePtr &data_type = nullptr);

  // brief Constructor
  //
  // param input [py::float_] the data for tensor
  // param data_type [TypeId] data type
  explicit Tensor(const py::float_ &input, const TypePtr &data_type = nullptr);

  // brief Constructor
  //
  // param input [py::int_] the data for tensor
  // param data_type [TypeId] data type
  explicit Tensor(const py::int_ &input, const TypePtr &data_type = nullptr);

  // brief Constructor
  //
  // param input [Tensor] the data for tensor
  // param data_type [TypeId] data type
  Tensor(const Tensor &tensor, const TypePtr &data_type = nullptr);

  ~Tensor() override = default;

  MS_DECLARE_PARENT(Tensor, MetaTensor);

  // brief Overloads operator = for Tensor.
  //
  // The constructed Tensor object has the same type and shape with tensor.
  //
  // param tensor An existing Tensor object.
  Tensor &operator=(const Tensor &tensor);

  // brief Compares two Tensor objects.
  //
  // Compare two tensor objects to see if they have same data type, shape and
  // data value.
  //
  // param tensor The Tensor object to be compared.
  // return true: If having same type, shape and data, return true, or return false.
  bool operator==(const Tensor &tensor) const;

  // It is different from 'operator==' which just compare shape/type/address, it do real value comparison.
  bool ValueEqual(const Tensor &other) const;

  bool operator==(const Value &other) const override {
    if (other.isa<Tensor>()) {
      auto other_ = static_cast<const Tensor &>(other);
      return *this == other_;
    } else {
      return false;
    }
  }

  py::tuple GetPyTupleShape() const;

  // brief Gets tensor's dimension
  //
  // return The number of dimensions of the tensor data.
  int DataDim() const;

  // brief Getting tensor data size
  //
  // return The total number of elements of the tensor data.
  int DataSize() const;

  // brief Tensor's data value.
  //
  // return [py::array] The tensor's data in py::array.
  py::array data() const;

  // brief Get the data type fo the tensor for C++
  //
  // return [int] The tensor's data type will be cast to int to return.
  int data_type_c() const;

  // brief Get the tensor's shape for C++
  //
  // return [std::vector<int>]
  std::vector<int> shape_c(void) const;

  // brief Get Tensor data pointer for c++ type
  //
  // param writable true if writable, false if read only
  // return The pointer to the object
  void *data_c(bool writable = false);

  // brief Get data type from tensor data.
  //
  // param buf The buffer info of the py::array data.
  // return The [TypeId] of the tensor data.
  TypeId GetDataType(const py::buffer_info &buf) const;

  // brief Sets the data type of a tensor.
  //
  // param data_type The data type of the tensor to be set.
  //
  TypeId set_data_type(const TypeId data_type) override;
  TypePtr SetDtype(const TypePtr type_ptr) override;
  std::string GetShapeAndDataTypeInfo() const;
  std::string ToString() const override;
  std::string ToStringRepr() const;
  py::array data_;  // < Tensor's data value
  const bool parse_info_ = true;
  bool is_init();
  void set_init_flag(bool flag);

 private:
  // brief init tensor
  //
  // param input [py::array] the data for tensor
  // param data_type [TypeId] data type
  // return true if succeed, false if failed.
  void init(const py::array &input, const TypeId &data_type);
  void init(const py::array &input, const TypePtr &type_ptr);
  bool init_flag_{false};
  // brief init tensor attribute
  //
  // param data_type [TypeId] Data type of the tensor.
  // param shape [py::array] The shape of the tensor.
  // return true if succeed, false if failed.
  void init(TypeId data_type, const std::vector<int> &shape, py::array *data);

  bool convert_data(const py::array &in, const TypeId in_data_type, py::array *out, const TypeId out_data_type);

 public:
  bool is_dirty() const { return dirty_; }
  void set_dirty(const bool dirty) { dirty_ = dirty; }
  DeviceAddressPtr device_address() const { return device_address_; }
  void set_device_address(const DeviceAddressPtr &device_address) { device_address_ = device_address; }
  py::array data_sync();
  std::string id() const { return id_; }

 private:
  bool dirty_{true};
  std::string id_{""};
  DeviceAddressPtr device_address_{nullptr};
};

using TensorPtr = std::shared_ptr<Tensor>;
using TensorPtrList = std::vector<std::shared_ptr<Tensor>>;

}  // namespace tensor
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_IR_TENSOR_H_
