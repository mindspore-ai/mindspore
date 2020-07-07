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
#include <numeric>

#include "Eigen/Core"
#include "device/device_address.h"
#include "ir/meta_tensor.h"
#include "include/ms_tensor.h"
#include "utils/log_adapter.h"

using float16 = Eigen::half;

using mindspore::device::DeviceAddress;
using DeviceAddressPtr = std::shared_ptr<mindspore::device::DeviceAddress>;
// brief mindspore namespace.
//
// mindspore namespace is the top level namespace of MindSpore project.
// Other namespace should be a sub namespace of mindspore namespace in the ME project.
namespace mindspore {
// brief mindspore::tensor namespace
//
// A sub namespace in ME to support tensor related definition.
namespace tensor {
// Tensor data interface.
class TensorData {
 public:
  /// Total number of elements.
  virtual ssize_t size() const = 0;
  /// Byte size of a single element.
  virtual ssize_t itemsize() const = 0;
  /// Total number of bytes.
  virtual ssize_t nbytes() const = 0;
  /// Number of dimensions.
  virtual ssize_t ndim() const = 0;
  /// Data pointer.
  virtual void *data() = 0;
  /// Is data equals.
  virtual bool equals(const TensorData &other) const = 0;
  /// To string.
  virtual std::string ToString() const = 0;
};

using TensorDataPtr = std::shared_ptr<TensorData>;

// Tensor entity class
class Tensor : public MetaTensor {
 public:
  abstract::AbstractBasePtr ToAbstract() override;

  // brief Create tensor from another tensor, data is shared.
  //
  // param tensor [Tensor] The input tensor.
  explicit Tensor(const Tensor &tensor);

  // brief Create tensor with given data type from another tensor.
  //
  // param tensor [Tensor] The input tensor.
  // param data_type [TypeId] The new tensor data type.
  Tensor(const Tensor &tensor, TypeId data_type);

  // brief Create tensor with the given shared tensor data.
  //
  // param data_type [TypeId] Data type of the tensor.
  // param shape The shape represented by std::vector<int> of the tensor.
  // param data The shared tensor data.
  Tensor(TypeId data_type, const std::vector<int> &shape, TensorDataPtr data);

  // brief Create an all zero tensor.
  //
  // param data_type [TypeId] Data type of the tensor.
  // param shape The shape represented by std::vector<int> of the tensor.
  Tensor(TypeId data_type, const std::vector<int> &shape);

  // brief Create a tensor with input data buffer.
  //
  // param data_type [TypeId] Data type of the tensor.
  // param shape The shape represented by std::vector<int> of the tensor.
  // param data The input data to be copied into tensor.
  // param data_len The length of data in bytes.
  Tensor(TypeId data_type, const std::vector<int> &shape, void *data, size_t data_len);

  // brief Create a tensor with input data buffer and given source data type.
  //
  // param data_type [TypeId] Data type of the tensor.
  // param shape The shape represented by std::vector<int> of the tensor.
  // param data The input data to be copied into tensor.
  // param src_data_type The source data type.
  Tensor(TypeId data_type, const std::vector<int> &shape, void *data, TypeId src_data_type);

  // brief Create 1 dimension tensor from an int vector.
  //
  // param input [std::vector<int64_t>] the data for tensor
  // param data_type [TypeId] data type
  explicit Tensor(const std::vector<int64_t> &input, const TypePtr &data_type = nullptr);

  // brief Create 1 dimension tensor from a float vector.
  //
  // param input [std::vector<double>] the data for tensor
  // param data_type [TypeId] data type
  explicit Tensor(const std::vector<double> &input, const TypePtr &data_type = nullptr);

  // brief Create 0 dimension tensor from an int scalar.
  //
  // param input [int64] the data for tensor
  // param data_type [TypeId] data type
  explicit Tensor(int64_t input, const TypePtr &data_type = nullptr);

  // brief Create 0 dimension tensor from a float scalar.
  //
  // param input [double] the data for tensor
  // param data_type [TypeId] data type
  explicit Tensor(double input, const TypePtr &data_type = nullptr);

  ~Tensor() override = default;

  MS_DECLARE_PARENT(Tensor, MetaTensor);

  // brief Compares two Tensor objects.
  //
  // Compare two tensor objects to see if they have same data type, shape and data address.
  //
  // param tensor The Tensor object to be compared.
  // return true: If having same type, shape and data address, return true, or return false.
  bool operator==(const Tensor &tensor) const;

  // It is different from 'operator==' which just compare shape/type/address,
  // it do real value comparison.
  bool ValueEqual(const Tensor &tensor) const;

  // assgin value to this tensor
  Tensor &AssignValue(const Tensor &tensor);

  bool operator==(const Value &other) const override {
    if (other.isa<Tensor>()) {
      auto &other_ = static_cast<const Tensor &>(other);
      return *this == other_;
    }
    return false;
  }

  // brief Gets tensor's dimension
  //
  // return The number of dimensions of the tensor data.
  int DataDim() const { return static_cast<int>(data().ndim()); }

  // brief Getting tensor data size
  //
  // return The total number of elements of the tensor data.
  int DataSize() const { return static_cast<int>(data().size()); }

  // brief Get the data type fo the tensor for C++
  //
  // return [int] The tensor's data type will be cast to int to return.
  int data_type_c() const { return static_cast<int>(data_type_); }

  // brief Get the tensor's shape for C++
  //
  // return [std::vector<int>]
  std::vector<int> shape_c(void) const { return shape(); }

  // brief Get Tensor data pointer for c++ type
  //
  // param writable true if writable, false if read only
  // return The pointer to the object
  void *data_c() { return data().data(); }

  // brief Get Tensor data byte-size for c++ type
  //
  // return byte size of Tensor data
  size_t Size() const { return data().nbytes(); }

  void *data_c() const { return data_->data(); }

  // brief Sync data with device.
  void data_sync() const;

  // brief Get the internal data object.
  //
  // return The reference to internal data object.
  TensorData &data() { return *data_; }

  // brief Get the internal data shared pointer.
  //
  // return The reference to internal data object.
  const TensorDataPtr &data_ptr() const { return data_; }

  // brief Get the internal data object.
  //
  // return The reference to internal data object.
  const TensorData &data() const { return *data_; }

  TypeId set_data_type(const TypeId data_type) override;

  std::string GetShapeAndDataTypeInfo() const;

  std::string ToString() const override;

  std::string ToStringRepr() const;

  bool is_init() { return init_flag_; }
  void set_init_flag(bool flag) { init_flag_ = flag; }

  bool is_dirty() const { return dirty_; }
  void set_dirty(const bool dirty) { dirty_ = dirty; }

  DeviceAddressPtr device_address() const { return device_address_; }
  void set_device_address(const DeviceAddressPtr &device_address) { device_address_ = device_address; }

  std::string id() const { return id_; }

  const bool parse_info_ = true;

 private:
  bool init_flag_{false};
  TensorDataPtr data_{nullptr};
  bool dirty_{true};
  std::string id_{""};
  DeviceAddressPtr device_address_{nullptr};
};
using TensorPtr = std::shared_ptr<Tensor>;
using TensorPtrList = std::vector<std::shared_ptr<Tensor>>;
}  // namespace tensor

namespace inference {
class Tensor : public MSTensor {
 public:
  Tensor(TypeId data_type, const std::vector<int> &shape);

  explicit Tensor(std::shared_ptr<tensor::Tensor> tensor_ptr);

  ~Tensor() = default;

  TypeId data_type() const override;

  TypeId set_data_type(const TypeId data_type) override;

  std::vector<int> shape() const override;

  size_t set_shape(const std::vector<int> &shape) override;

  int DimensionSize(size_t index) const override;

  int ElementsNum() const override;

  std::size_t hash() const override;

  std::shared_ptr<tensor::Tensor> tensor() const;

  size_t Size() const override;

  void *MutableData() const override;

 protected:
  std::shared_ptr<tensor::Tensor> tensor_impl_;
};
}  // namespace inference
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_IR_TENSOR_H_
