/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_MAP_TENSOR_H_
#define MINDSPORE_CORE_IR_MAP_TENSOR_H_

#include <tuple>
#include <memory>
#include <string>
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/tensor.h"
#include "ir/param_info.h"
#include "utils/macros.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace tensor {
class MapTensor;
// Smart pointer for MapTensor.
using MapTensorPtr = std::shared_ptr<MapTensor>;
///
/// \brief MapTensor is a dynamic tensor with map like index functions.
///
class MS_CORE_API MapTensor final : public Tensor {
 public:
  struct ExportData {
    TensorPtr key_tensor;
    TensorPtr value_tensor;
    TensorPtr status_tensor;
  };

  enum class Status {
    kUnchanged = 0,
    kModified = 1,
    kErased = 2,
  };

  /// \brief Create a empty MapTensor.
  ///
  /// \param[in] key_dtype [TypeId] The key data type id.
  /// \param[in] value_dtype [TypeId] The value data type id.
  /// \param[in] value_shape [TypeId] The value shape.
  /// \param[in] default_value [ValuePtr] the default value.
  MapTensor(TypeId key_dtype, TypeId value_dtype, const ShapeVector &value_shape, const ValuePtr &default_value)
      : key_dtype_(key_dtype), default_value_(default_value) {
    data_type_ = value_dtype;
    value_shape_ = value_shape;
    shape_ = {abstract::Shape::kShapeDimAny};
    (void)shape_.insert(shape_.end(), value_shape.begin(), value_shape.end());
    ShapeVector key_shape = {abstract::Shape::kShapeDimAny};
    key_tensor_ = std::make_shared<Tensor>(key_dtype, key_shape);
    value_tensor_ = std::make_shared<Tensor>(value_dtype, value_shape);
    status_tensor_ = std::make_shared<Tensor>(kNumberTypeUInt8, key_shape);
  }

  ~MapTensor() override = default;

  MS_DECLARE_PARENT(MapTensor, Tensor)

  std::size_t hash() const override;

  bool operator==(const Value &other) const override {
    if (this == &other) {
      return true;
    }
    if (!other.isa<MapTensor>()) {
      return false;
    }
    auto other_ = static_cast<const MapTensor &>(other);
    return *this == other_;
  }

  bool operator==(const MapTensor &other) const;

  TypeId key_dtype() const { return key_dtype_; }

  TypeId value_dtype() const { return data_type_; }

  size_t size() const { return size_; }

  const ShapeVector &value_shape() const { return value_shape_; }

  const ValuePtr &default_value() const { return default_value_; }

  TypePtr KeyDtype() const { return TypeIdToType(key_dtype_); }

  TypePtr ValueDtype() const { return TypeIdToType(data_type_); }

  abstract::AbstractBasePtr ToAbstract() override;

  std::string ToString() const override;

  /// \brief Get or create values.
  ///
  /// \param[in] key_tensor [Tensor] The key tensor.
  /// \return The value tensor according the key tensor, return default_value if key_tensor is not exist.
  TensorPtr Get(const TensorPtr &key_tensor);

  /// \brief Put or insert key value pairs.
  ///
  /// \param[in] key_tensor [Tensor] The key tensor.
  /// \param[in] value_tensor [Tensor] The value tensor.
  void Put(const TensorPtr &key_tensor, const TensorPtr &value_tensor);

  /// \brief Remove items with the given keys.
  ///
  /// \param[in] key_tensor [Tensor] The key tensor.
  void Erase(const TensorPtr &key_tensor);

  /// \brief Update MapTensor from exported data.
  ///
  /// \param[in] data [ExportData] The data.
  void Update(const ExportData &data);

  /// \brief Update MapTensor from exported data.
  ///
  /// \param[in] full [bool] True for full export, false for incremental export.
  /// \return The exported data.
  ExportData Export(bool full = false);

  /// \brief Get the key tensor of MapTensor data.
  ///
  /// \return The key tensor.
  const TensorPtr &KeyTensor() const { return key_tensor_; }

  /// \brief Get the value tensor of MapTensor data.
  ///
  /// \return The value tensor.
  const TensorPtr &ValueTensor() const { return value_tensor_; }

  /// \brief Get the status tensor of MapTensor data.
  ///
  /// \return The status tensor.
  const TensorPtr &StatusTensor() const { return status_tensor_; }

 private:
  // Data type of the key.
  TypeId key_dtype_;

  // Default value. should be a scalar as the initial value or a string as the initializer name.
  ValuePtr default_value_;

  // the shape of values
  ShapeVector value_shape_;

  // the size of keys, shape_ is (size_, value_shape_).
  size_t size_;

  // Key tensor of data.
  TensorPtr key_tensor_;

  // Value tensor of data.
  TensorPtr value_tensor_;

  // Status tensor of data.
  TensorPtr status_tensor_;
};
}  // namespace tensor
}  // namespace mindspore
#endif  // MINDSPORE_CORE_IR_MAP_TENSOR_H_
