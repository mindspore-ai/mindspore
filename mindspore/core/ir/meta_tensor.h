/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_META_TENSOR_H_
#define MINDSPORE_CORE_IR_META_TENSOR_H_

#include <utility>
#include <vector>
#include <memory>
#include <string>

#include "base/base.h"
#include "ir/param_info.h"
#include "ir/dtype.h"
#include "utils/convert_utils_base.h"
#include "utils/hashing.h"
#include "utils/shape_utils.h"

// brief mindspore namespace.
//
// mindspore namespace is the top level namespace of MindSpore project.
// Other namespace should be a sub namespace of mindspore namespace in the ME project.
namespace mindspore {

// brief mindspore::tensor namespace
//
// A sub namespace in ME to support tensor related definition.
namespace tensor {
// brief Device info of Tensor
//
// Includes the format, data type and host format of a tensor.
struct DeviceInfo {
  explicit DeviceInfo(std::string format = "DefaultFormat", TypePtr data_type = nullptr,
                      std::string host_format = "DefaultFormat")
      : format_(std::move(format)), data_type_(std::move(data_type)), host_format_(std::move(host_format)) {}
  std::string format_ = "DefaultFormat";
  TypePtr data_type_ = nullptr;
  std::string host_format_ = "DefaultFormat";
};

// brief Metadata of Tensor
//
// Includes the metadata information of a tensor, such as data type, shape
// and so on. But it does not contain values of a tensor.
class MS_CORE_API MetaTensor : public Value {
 public:
  /// \brief Construction
  MetaTensor();

  /// \brief Constructs a meta tensor of a tensor having data_type data and shape.
  /// The constructed MetaTensor is not a Tensor, but it has the data type and shape
  /// information of a Tensor.
  ///
  /// \param[in] data_type The data type of the tensor.
  /// \param[in] shape The shape of the tensor.
  MetaTensor(TypeId data_type, const ShapeVector &shape);

  MetaTensor(const TypePtr &type_ptr, const ShapeVector &shape);
  /// \brief Copy constructor.
  /// The constructed MetaTensor object will have the same data type and shape as the
  /// meta_tensor.
  ///
  /// \param[in] meta_tensor An existing MetaTensor object.
  MetaTensor(const MetaTensor &meta_tensor);

  /// \brief Destrustor of MetaTensor.
  ~MetaTensor() override = default;
  MS_DECLARE_PARENT(MetaTensor, Value)

  /// \brief Overloads operator = for MetaTensor.
  /// The constructed MetaTensor object has the same type and shape with meta_tensor.
  ///
  /// \param[in] meta_tensor An existing MetaTensor object.
  /// \return A MetaTensor object.
  virtual MetaTensor &operator=(const MetaTensor &meta_tensor);

  /// \brief Compares two MetaTensor objects.
  /// The constructed MetaTensor object has the same type and shape with meta_tensor.
  ///
  /// \param[in] meta_tensor The MetaTensor object to be compared.
  /// \return Return true if having same type and shape, otherwise return false.
  virtual bool operator==(const MetaTensor &meta_tensor) const;

  /// \brief Get the data type of the tensor in its MetaTensor.
  /// All the types are defined in "ir/dtype.h".
  ///
  /// \return The data type of the tensor in its MetaTensor.
  TypePtr Dtype() const;

  abstract::AbstractBasePtr ToAbstract() override;

  /// \brief Get the data type of a tensor in its MetaTensor.
  ///
  /// \return The data type.
  TypeId data_type() const { return data_type_; }

  std::string ToString() const override;
  std::string DumpText() const override;

  /// \brief Set the data type of a tensor in its MetaTensor.
  ///
  /// \param[in] data_type The data type of the tensor to be set.
  virtual TypeId set_data_type(TypeId data_type) {
    data_type_ = data_type;
    return data_type_;
  }

  /// \brief Set the dtype of a tensor in its MetaTensor.
  ///
  /// \param[in] type_ptr The dtype of the tensor to be set.
  virtual TypePtr SetDtype(const TypePtr type_ptr);

  /// \brief Get tensor's shape.
  /// The shape of a tensor is stored in a vector<int>. Each
  /// element of the vector represents the size of a dimension of the tensor.
  /// The order of each element in the vector is the same as the the dimension's
  /// order it represents.
  ///
  /// \return A const vector<int> which represents the shape of the tensor.
  const ShapeVector &shape() const { return shape_; }

  /// \brief Sets the shape of a tensor.
  /// The shape of a tensor is stored in a vector<int>. Each
  /// element of the vector represents the size of a dimension of the tensor.
  /// The order of each element in the vector is the same as the the dimension's
  /// order it represents.
  ///
  /// \param[in] shape The shape of the tensor.
  /// \return The shape's size.
  virtual size_t set_shape(const ShapeVector &shape) {
    this->shape_ = shape;
    return shape_.size();
  }

  /// \brief Get tensor's device info.
  ///
  /// \return The device info of this tensor.
  DeviceInfo device_info() const { return device_info_; }

  /// \brief Set tensor's device info.
  ///
  /// \param[in] device_info The tensor's device info.
  void set_device_info(const DeviceInfo &device_info) { device_info_ = device_info; }

  /// \brief Set tensor's device info.
  ///
  /// \param[in] format The input format.
  /// \param[in] data_type The input data type.
  /// \param[in] host_format The input host format.
  void SetDeviceInfo(const std::string &format, const TypePtr &data_type,
                     const std::string &host_format = "DefaultFormat");

  /// \brief Get the size of a given dimension by its index number.
  ///
  /// \return The size of a given dimension by its index number.
  int64_t DimensionSize(size_t index) const;

  /// \brief Get total number of elements in a tensor.
  ///
  /// \return The total number of elements in a tensor.
  int ElementsNum() const;

  std::size_t hash() const override {
    std::size_t hash_value = std::hash<int>{}(static_cast<int>(data_type_));
    hash_value = hash_combine(hash_value, std::hash<size_t>{}(shape_.size()));
    // hash all elements may costly, so only take at most 4 elements into account based on
    // some experiments.
    for (size_t i = 0; (i < shape_.size()) && (i < 4); ++i) {
      hash_value = hash_combine(hash_value, (std::hash<int>{}(LongToInt(shape_[i]))));
    }
    return hash_value;
  }
  bool operator==(const Value &other) const override {
    if (other.isa<MetaTensor>()) {
      auto other_ = static_cast<const MetaTensor &>(other);
      return *this == other_;
    } else {
      return false;
    }
  }
  /// \brief Get tensor's param_info info.
  ///
  /// \return The tensor's param_info info.
  ParamInfoPtr param_info() const { return param_info_; }

  /// \brief Check whether this Tensor is a parameter.
  ///
  /// \return Whether this Tensor is a parameter.
  bool is_parameter() const { return is_parameter_; }

  /// \brief Set tensor's param_info info.
  ///
  /// \param[in] param_info The input param_info.
  void set_param_info(const ParamInfoPtr &param_info) {
    is_parameter_ = true;
    param_info_ = param_info;
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
  // A ShapeVector container is used to store the shape of a tensor.
  // Each element of the vector represents the size of a dimension of the tensor.
  // The order of each element in the vector is as same as the the dimension's
  // order it represents. If the dimension size is not set, its value will be -1.
  ShapeVector shape_;

  // brief Device info of Tensor
  //
  // Includes the format and data type of a tensor on device.
  DeviceInfo device_info_;

  bool is_parameter_{false};
  ParamInfoPtr param_info_{nullptr};
};

using MetaTensorPtr = std::shared_ptr<MetaTensor>;

// brief Metadata of SparseTensor
//
// Includes the metadata information of a SparseTensor, such as data type, shape
// and so on. But it does not contain values of a SparseTensor.
class MS_CORE_API MetaSparseTensor : public Value {
 public:
  /// \brief Construction
  MetaSparseTensor();

  /// \brief Constructs a meta SparseTensor having data_type data and shape.
  /// The constructed MetaSparseTensor contains the data type and shape information of
  /// a SparseTensor.
  ///
  /// \param[in] data_type The data type of the SparseTensor.
  /// \param[in] shape The shape of the SparseTensor.
  MetaSparseTensor(TypeId data_type, const ShapeVector &shape);

  /// \brief Copy constructor.
  /// The constructed MetaSparseTensor object will have the same data type and shape as the
  /// meta_sparse_tensor.
  ///
  /// \param[in] meta_tensor An existing MetaSparseTensor object.
  MetaSparseTensor(const MetaSparseTensor &meta_sparse_tensor);

  /// \brief Copy assignment operator.
  ///
  /// \param[in] meta_sparse_tensor An existing MetaSparseTensor object.
  /// \return A MetaSparseTensor object set with the same data type and shape as the meta_sparse_tensor.
  MetaSparseTensor &operator=(const MetaSparseTensor &meta_sparse_tensor);

  /// \brief Destrustor of MetaSparseTensor.
  ~MetaSparseTensor() override = default;
  MS_DECLARE_PARENT(MetaSparseTensor, Value)

  /// \brief Compares two MetaSparseTensor objects.
  /// The constructed MetaSparseTensor object has the same type and shape with meta_sparse_tensor.
  ///
  /// \param[in] meta_sparse_tensor The MetaSparseTensor object to be compared.
  /// \return Return true if having same type and shape, otherwise return false.
  virtual bool operator==(const MetaSparseTensor &meta_sparse_tensor) const {
    return data_type_ == meta_sparse_tensor.data_type() && shape_ == meta_sparse_tensor.shape();
  }

  /// \brief Get the data type of the sparse tensor.
  /// All the types are defined in "ir/dtype.h".
  ///
  /// \return The data type of the sparse tensor.
  TypePtr Dtype() const;

  /// \brief Get the data type of a sparse tensor.
  ///
  /// \return The data type.
  TypeId data_type() const { return data_type_; }

  /// \brief Set the data type of a sparse tensor.
  ///
  /// \param[in] data_type The data type of the tensor to be set.
  void set_data_type(TypeId data_type) { data_type_ = data_type; }

  /// \brief Get sparsetensor's shape.
  ///
  /// \return A const vector<int> which represents the shape of the tensor.
  const ShapeVector &shape() const { return shape_; }

  /// \brief Sets the shape of a sparsetensor.
  ///
  /// \param[in] shape The shape of the tensor.
  void set_shape(const ShapeVector &shape) { this->shape_ = shape; }

  /// \brief Get display information of this Tensor.
  ///
  /// \return The display information of this Tensor.
  virtual std::string ToString() const = 0;

 protected:
  // Data type of the sparsetensor.
  TypeId data_type_;

  // Shape of the sparsetensor.
  ShapeVector shape_;
};

using MetaSparseTensorPtr = std::shared_ptr<MetaSparseTensor>;
}  // namespace tensor
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_META_TENSOR_H_
