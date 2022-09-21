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
///
/// \brief MapTensor is a dynamic tensor with map like index functions.
///
class MS_CORE_API MapTensor final : public Value {
 public:
  using Tensor = tensor::Tensor;
  using TensorPtr = tensor::TensorPtr;

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
  MapTensor(TypeId key_dtype, TypeId value_dtype, const ShapeVector &value_shape)
      : key_dtype_(key_dtype), value_dtype_(value_dtype), value_shape_(value_shape) {}

  ~MapTensor() override = default;

  MS_DECLARE_PARENT(MapTensor, Value)

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

  TypeId value_dtype() const { return value_dtype_; }

  const ShapeVector &value_shape() const { return value_shape_; }

  TypePtr KeyDtype() const { return TypeIdToType(key_dtype_); }

  TypePtr ValueDtype() const { return TypeIdToType(value_dtype_); }

  abstract::AbstractBasePtr ToAbstract() override;

  std::string ToString() const override {
    auto key_dtype = KeyDtype();
    auto value_dtype = ValueDtype();
    return "MapTensor(key_dtype=" + (key_dtype == nullptr ? "<null>" : key_dtype->ToString()) +
           ", value_dtype=" + (value_dtype == nullptr ? "<null>" : value_dtype->ToString()) +
           ", value_shape=" + tensor::ShapeToString(value_shape_) + ")";
  }

  /// \brief Get tensor's param_info info.
  ///
  /// \return The tensor's param_info info.
  const ParamInfoPtr &param_info() const { return param_info_; }

  /// \brief Check whether this MapTensor is a parameter.
  ///
  /// \return Whether this MapTensor is a parameter.
  bool is_parameter() const { return param_info_ != nullptr; }

  /// \brief Set param_info info.
  ///
  /// \param[in] param_info The input param_info.
  void set_param_info(const ParamInfoPtr &param_info) { param_info_ = param_info; }

  /// \brief Get or create values.
  ///
  /// \param[in] key_tensor [Tensor] The key tensor.
  /// \param[in] default_value [Tensor] The default value tensor.
  /// \return The value tensor according the key tensor.
  TensorPtr Get(const TensorPtr &key_tensor, const TensorPtr &default_value);

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

 private:
  // Data type of the key.
  TypeId key_dtype_;

  // Data type of the value.
  TypeId value_dtype_;

  // Shape of the value.
  ShapeVector value_shape_;

  // Parameter information.
  ParamInfoPtr param_info_;
};

// Smart pointer for MapTensor.
using MapTensorPtr = std::shared_ptr<MapTensor>;
}  // namespace mindspore
#endif  // MINDSPORE_CORE_IR_MAP_TENSOR_H_
