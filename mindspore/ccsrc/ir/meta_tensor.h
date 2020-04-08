/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_IR_META_TENSOR_H_
#define MINDSPORE_CCSRC_IR_META_TENSOR_H_

#include <utility>
#include <vector>
#include <memory>
#include <string>
#include "device/device_address.h"

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

#include "Eigen/Core"
#include "ir/base.h"
#include "ir/dtype.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils.h"
#include "utils/hashing.h"

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

// brief Device info of Tensor
//
// Includes the format and data type of a tensor.
struct DeviceInfo {
  explicit DeviceInfo(std::string format = "DefaultFormat", TypePtr data_type = nullptr)
      : format_(std::move(format)), data_type_(std::move(data_type)) {}
  std::string format_ = "DefaultFormat";
  TypePtr data_type_ = nullptr;
};

// brief Metadata of Tensor
//
// Includes the metadata information of a tensor, such as data type, shape
// and so on. But it does not contain values of a tensor.
class MetaTensor : public Value {
 public:
  // Construction
  MetaTensor();

  // brief Constructs a meta tensor of a tensor having data_type data and shape.
  //
  // The constructed MetaTensor is not a Tensor, but it has the data type and shape
  // information of a Tensor. The following codes will create a 2x3 float
  // param data_type The data type of the tensor.
  // param shape The shape of the tensor.
  MetaTensor(const TypeId data_type, const std::vector<int>& shape);

  MetaTensor(const TypePtr& type_ptr, const py::tuple& shape);
  // brief Constructs a MetaTensor object from an existing MetaTensor instance.
  //
  // The constructed MetaTensor object will have the same data type and shape as the
  // meta_tensor.
  //
  // param meta_tensor An existing MetaTensor object.
  MetaTensor(const MetaTensor& meta_tensor);
  ~MetaTensor() override = default;
  MS_DECLARE_PARENT(MetaTensor, Value)

  // brief Overloads operator = for MetaTensor.
  //
  // The constructed MetaTensor object has the same type and shape with meta_tensor.
  //
  // param meta_tensor An existing MetaTensor object.
  virtual MetaTensor& operator=(const MetaTensor& meta_tensor);

  // brief Compares two MetaTensor objects.
  //
  // The constructed MetaTensor object has the same type and shape with meta_tensor.
  //
  // param meta_tensor The MetaTensor object to be compared.
  // return true: If having same type and shape, return true, or return false.
  virtual bool operator==(const MetaTensor& meta_tensor) const;

  // brief Returns the data type of the tensor in its MetaTensor.
  //
  // All the types are defined in "ir/dtype.h".
  TypePtr Dtype() const;
  TypeId data_type() const { return data_type_; }
  std::string ToString() const override;
  std::string DumpText() const override;
  // brief Sets the data type of a tensor in its MetaTensor.
  //
  // param data_type The data type of the tensor to be set.
  virtual TypeId set_data_type(const TypeId data_type) {
    data_type_ = data_type;
    return data_type_;
  }
  virtual TypePtr SetDtype(const TypePtr type_ptr);
  // brief Get tensor's shape.
  //
  // The shape of a tensor is stored in a vector<int>. Each
  // element of the vector represents the size of a dimension of the tensor.
  // The order of each element in the vector is as same as the the dimension's
  // order it represents.
  //
  // return A const vector<int> which represents the shape of the tensor.
  std::vector<int> shape() const { return shape_; }

  // brief Sets the shape of a tensor.
  //
  // The shape of a tensor is stored in a vector<int>. Each
  // element of the vector represents the size of a dimension of the tensor.
  // The order of each element in the vector is as same as the the dimension's
  // order it represents.
  //
  // param shape The shape of the tensor.
  // return The shape's size.
  size_t set_shape(const std::vector<int>& shape) {
    this->shape_ = shape;
    return shape_.size();
  }

  // Get tensor's device info.
  DeviceInfo device_info() const { return device_info_; }

  // Set tensor's device info.
  void set_device_info(const DeviceInfo& device_info) { device_info_ = device_info; }

  void SetDeviceInfo(const std::string& format, const TypePtr& data_type);

  // Get the size of a given dimension by its index number.
  int DimensionSize(size_t index) const;

  // Get total number of elements in a tensor.
  int ElementsNum() const;

  std::size_t hash() const override {
    std::size_t hash_value = std::hash<int>{}(SizeToInt(data_type_));
    hash_value = hash_combine(hash_value, std::hash<size_t>{}(shape_.size()));
    // hash all elements may costly, so only take at most 4 elements into account based on
    // some experiments.
    for (size_t i = 0; (i < shape_.size()) && (i < 4); ++i) {
      hash_value = hash_combine(hash_value, (std::hash<int>{}(shape_[i])));
    }
    return hash_value;
  }
  bool operator==(const Value& other) const override {
    if (other.isa<MetaTensor>()) {
      auto other_ = static_cast<const MetaTensor&>(other);
      return *this == other_;
    } else {
      return false;
    }
  }

 protected:
  // brief Data type of the tensor.
  //
  // All support data type is in Number Types of [TypeId],
  // including [kNumberTypeBool], [kNumberTypeInt],
  // [kNumberTypeUInt32], [kNumberTypeFloat32] and [kNumberTypeFloat64].
  TypeId data_type_;

  // brief Shape of the tensor.
  //
  // A std::vector<int> container is used to store the shape of a tensor.
  // Each element of the vector represents the size of a dimension of the tensor.
  // The order of each element in the vector is as same as the the dimension's
  // order it represents. If the dimension size is not set, its value will be -1.
  std::vector<int> shape_;

  // brief Device info of Tensor
  //
  // Includes the format and data type of a tensor on device.
  DeviceInfo device_info_;
};

// Tensor entity class
class Tensor : public MetaTensor {
 public:
  Tensor() = default;
  abstract::AbstractBasePtr ToAbstract() override;
  // brief Constructor for Python.
  //
  // param type_ptr [TypePty] Data type of the tensor.
  // param py_shape [py::tuple] The shape represented by py::tuple of the tensor.
  Tensor(const TypePtr& type_ptr, const py::tuple& shape);

  // brief Constructor for C++.
  //
  // param data_type [TypeId] Data type of the tensor.
  // param shape The shape represented by std::vector<int> of the tensor.
  Tensor(TypeId data_type, const std::vector<int>& shape);

  // brief Constructor for Python.
  //
  // param input [py::array] Data value of the tensor.
  // param data_type [TypeId] Data type of the tensor.
  explicit Tensor(const py::array& input, const TypePtr& data_type = nullptr);

  // brief Constructor
  //
  // param input [py::list] the data for tensor
  // param data_type [TypeId] data type
  explicit Tensor(const py::list& input, const TypePtr& data_type = nullptr);

  // brief Constructor
  //
  // param input [py::tuple] the data for tensor
  // param data_type [TypeId] data type
  explicit Tensor(const py::tuple& input, const TypePtr& data_type = nullptr);

  // brief Constructor
  //
  // param input [py::float_] the data for tensor
  // param data_type [TypeId] data type
  explicit Tensor(const py::float_& input, const TypePtr& data_type = nullptr);

  // brief Constructor
  //
  // param input [py::int_] the data for tensor
  // param data_type [TypeId] data type
  explicit Tensor(const py::int_& input, const TypePtr& data_type = nullptr);

  // brief Constructor
  //
  // param input [Tensor] the data for tensor
  // param data_type [TypeId] data type
  Tensor(const Tensor& tensor, const TypePtr& data_type = nullptr);

  ~Tensor() override = default;

  MS_DECLARE_PARENT(Tensor, MetaTensor);

  // brief Overloads operator = for Tensor.
  //
  // The constructed Tensor object has the same type and shape with tensor.
  //
  // param tensor An existing Tensor object.
  Tensor& operator=(const Tensor& tensor);

  // brief Compares two Tensor objects.
  //
  // Compare two tensor objects to see if they have same data type, shape and
  // data value.
  //
  // param tensor The Tensor object to be compared.
  // return true: If having same type, shape and data, return true, or return false.
  bool operator==(const Tensor& tensor) const;

  // It is different from 'operator==' which just compare shape/type/address, it do real value comparison.
  bool ValueEqual(const Tensor& other) const;

  // It is different from 'operator==' which just compare shape/type/address, it do real value comparison.
  bool ValueEqualPy(const py::object& other) const;

  bool operator==(const Value& other) const override {
    if (other.isa<Tensor>()) {
      auto other_ = static_cast<const Tensor&>(other);
      return *this == other_;
    } else {
      return false;
    }
  }

  // brief Gets tensor's dimension
  //
  // return The number of dimensions of the tensor data.
  int DataDim() const;

  // brief Getting tensor data size
  //
  // return The total number of elements of the tensor data.
  int DataSize() const;

  // brief Get tensor's shape
  //
  // return [py::tuple] The tensor's shape
  py::tuple GetPyTupleShape() const;

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
  void* data_c(bool writable = false);

  // brief Get data type from tensor data.
  //
  // param buf The buffer info of the py::array data.
  // return The [TypeId] of the tensor data.
  TypeId GetDataType(const py::buffer_info& buf) const;

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

 private:
  // brief init tensor
  //
  // param input [py::array] the data for tensor
  // param data_type [TypeId] data type
  // return true if succeed, false if failed.
  void init(const py::array& input, const TypeId& data_type);
  void init(const py::array& input, const TypePtr& type_ptr);

  // brief init tensor attribute
  //
  // param data_type [TypeId] Data type of the tensor.
  // param shape [py::array] The shape of the tensor.
  // return true if succeed, false if failed.
  void init(TypeId data_type, const std::vector<int>& shape, py::array* data);

  bool convert_data(const py::array& in, const TypeId in_data_type, py::array* out, const TypeId out_data_type);

 public:
  bool is_dirty() const { return dirty_; }
  void set_dirty(const bool dirty) { dirty_ = dirty; }
  DeviceAddressPtr device_address() const { return device_address_; }
  void set_device_address(const DeviceAddressPtr& device_address) { device_address_ = device_address; }
  py::array data_sync();

 private:
  bool dirty_{true};
  DeviceAddressPtr device_address_{nullptr};
};

using TensorPtr = std::shared_ptr<Tensor>;
using TensorPtrList = std::vector<std::shared_ptr<Tensor>>;

}  // namespace tensor
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_IR_META_TENSOR_H_
