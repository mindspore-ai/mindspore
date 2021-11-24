/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_DTYPE_TENSORTYPE_H_
#define MINDSPORE_CORE_IR_DTYPE_TENSORTYPE_H_

#include <cstddef>
#include <iostream>
#include <initializer_list>
#include <map>
#include <memory>
#include <utility>
#include <sstream>
#include <string>
#include <vector>
#include <type_traits>
#include <algorithm>

#include "utils/hash_map.h"
#include "base/base.h"
#include "ir/named.h"
#include "ir/dtype/type.h"

namespace mindspore {
/// \brief UndeterminedType defines interface for tensor undetermined data type.
class MS_CORE_API UndeterminedType final : public Object {
 public:
  /// \brief Default constructor for UndeterminedType.
  UndeterminedType() : Object(kObjectTypeUndeterminedType) {}

  /// \brief Constructor for UndeterminedType.
  ///
  /// \param[in] ele The element of UndeterminedType.
  explicit UndeterminedType(const TypePtr &ele)
      : Object(kObjectTypeUndeterminedType, kMetaTypeObject, false), element_type_(ele) {}

  /// \brief Destructor of UndeterminedType.
  ~UndeterminedType() override = default;
  MS_DECLARE_PARENT(UndeterminedType, Object)

  TypeId generic_type_id() const override { return kObjectTypeUndeterminedType; }

  /// \brief Get the element of UndeterminedType object.
  ///
  /// \return The element of UndeterminedType object.
  const TypePtr element() const { return element_type_; }

  /// \brief Set the element of UndeterminedType object.
  ///
  /// \param[in] element_type Define the element type to be set.
  void set_element(const TypePtr &element_type) { element_type_ = element_type; }

  TypePtr DeepCopy() const override;
  std::string ToString() const override;
  std::string ToReprString() const override;
  std::string DumpText() const override;

  bool operator==(const Type &other) const override;

 protected:
  TypePtr element_type_;
};
using MetaTensorTypePtr = std::shared_ptr<UndeterminedType>;

/// \brief TensorType defines interface for tensor data type.
class MS_CORE_API TensorType : public Object {
 public:
  /// \brief Default constructor for TensorType.
  TensorType() : Object(kObjectTypeTensorType, kObjectTypeUndeterminedType) {}

  /// \brief Constructor for TensorType.
  ///
  /// \param[in] ele The element of TensorType.
  explicit TensorType(const TypePtr &ele)
      : Object(kObjectTypeTensorType, kObjectTypeUndeterminedType, false), element_type_(ele) {}

  /// \brief Destructor of TensorType.
  ~TensorType() override = default;
  MS_DECLARE_PARENT(TensorType, Object)

  TypeId generic_type_id() const override { return kObjectTypeTensorType; }

  /// \brief Get the element of TensorType object.
  ///
  /// \return The element of TensorType object.
  const TypePtr element() const { return element_type_; }

  /// \brief Set the element of TensorType object.
  ///
  /// \param[in] element_type Define the element type to be set.
  void set_element(const TypePtr &element_type) { element_type_ = element_type; }

  TypePtr DeepCopy() const override;
  std::string ToString() const override;
  std::string ToReprString() const override;
  std::string DumpText() const override;
  bool operator==(const Type &other) const override;

 private:
  TypePtr element_type_;
};
using TensorTypePtr = std::shared_ptr<TensorType>;

/// \brief RowTensorType defines interface for row tensor data type.
class MS_CORE_API RowTensorType final : public Object {
 public:
  /// \brief Default constructor for RowTensorType.
  RowTensorType() : Object(kObjectTypeRowTensorType, kObjectTypeUndeterminedType) {}

  /// \brief Constructor for RowTensorType.
  ///
  /// \param[in] ele The element of RowTensorType.
  explicit RowTensorType(const TypePtr &ele)
      : Object(kObjectTypeRowTensorType, kObjectTypeUndeterminedType, false), element_type_(ele) {}

  /// \brief Destructor of RowTensorType.
  ~RowTensorType() override = default;
  MS_DECLARE_PARENT(RowTensorType, Object)

  TypeId generic_type_id() const override { return kObjectTypeRowTensorType; }

  /// \brief Get the element of RowTensorType object.
  ///
  /// \return The element of RowTensorType object.
  const TypePtr element() const { return element_type_; }

  /// \brief Set the element of RowTensorType object.
  ///
  /// \param[in] element_type Define the element type to be set.
  void set_element(const TypePtr &element_type) { element_type_ = element_type; }

  TypePtr DeepCopy() const override;
  std::string ToString() const override;
  std::string ToReprString() const override;
  std::string DumpText() const override;
  bool operator==(const Type &other) const override;

 private:
  TypePtr element_type_;
};
using RowTensorTypePtr = std::shared_ptr<RowTensorType>;

/// \brief SparseTensorType defines interface for sparse tensor data type.
class MS_CORE_API SparseTensorType final : public Object {
 public:
  /// \brief Default constructor for SparseTensorType.
  SparseTensorType() : Object(kObjectTypeSparseTensorType, kObjectTypeUndeterminedType) {}

  /// \brief Constructor for SparseTensorType.
  ///
  /// \param[in] ele The element of SparseTensorType.
  explicit SparseTensorType(const TypePtr &ele)
      : Object(kObjectTypeSparseTensorType, kObjectTypeUndeterminedType, false), element_type_(ele) {}

  /// \brief Destructor of SparseTensorType.
  ~SparseTensorType() override = default;
  MS_DECLARE_PARENT(SparseTensorType, Object)

  TypeId generic_type_id() const override { return kObjectTypeSparseTensorType; }

  /// \brief Get the element of SparseTensorType object.
  ///
  /// \return The element of SparseTensorType object.
  const TypePtr element() const { return element_type_; }

  /// \brief Set the element of SparseTensorType object.
  ///
  /// \param[in] element_type Define the element type to be set.
  void set_element(const TypePtr &element_type) { element_type_ = element_type; }

  TypePtr DeepCopy() const override;
  std::string ToString() const override;
  std::string ToReprString() const override;
  std::string DumpText() const override;
  bool operator==(const Type &other) const override;

 private:
  TypePtr element_type_;
};
using SparseTensorTypePtr = std::shared_ptr<SparseTensorType>;

/// \brief CSRTensorType defines interface for sparse tensor data type.
class MS_CORE_API CSRTensorType : public Object {
 public:
  /// \brief Default constructor for CSRTensorType.
  CSRTensorType() : Object(kObjectTypeCSRTensorType, kObjectTypeUndeterminedType) {}

  /// \brief Constructor for CSRTensorType.
  ///
  /// \param[in] ele The element of CSRTensorType.
  explicit CSRTensorType(const TypePtr &ele)
      : Object(kObjectTypeCSRTensorType, kObjectTypeUndeterminedType, false), element_type_(ele) {}

  /// \brief Destructor of CSRTensorType.
  ~CSRTensorType() override = default;
  MS_DECLARE_PARENT(CSRTensorType, Object)

  TypeId generic_type_id() const override { return kObjectTypeCSRTensorType; }

  /// \brief Get the element of CSRTensorType object.
  ///
  /// \return The element of CSRTensorType object.
  const TypePtr element() const { return element_type_; }

  /// \brief Set the element of CSRTensorType object.
  ///
  /// \param[in] element_type Define the element type to be set.
  void set_element(const TypePtr &element_type) { element_type_ = element_type; }

  TypePtr DeepCopy() const override;
  std::string ToString() const override;
  std::string ToReprString() const override;
  std::string DumpText() const override;
  bool operator==(const Type &other) const override;

 private:
  TypePtr element_type_;
};
using CSRTensorTypePtr = std::shared_ptr<CSRTensorType>;
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_DTYPE_TENSORTYPE_H_
