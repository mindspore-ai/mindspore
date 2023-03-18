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
  std::size_t hash() const override;

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
  size_t hash() const override;

 private:
  TypePtr element_type_;
};
using TensorTypePtr = std::shared_ptr<TensorType>;

/// \brief SparseTensorType is the base type for all sparse tensors.
class MS_CORE_API SparseTensorType : public Object {
 public:
  SparseTensorType() : Object(kObjectTypeSparseTensorType, kObjectTypeUndeterminedType) {}

  explicit SparseTensorType(const TypeId object_type) : Object(object_type, kObjectTypeUndeterminedType) {}

  explicit SparseTensorType(const TypePtrList &objs)
      : Object(kObjectTypeSparseTensorType, kObjectTypeUndeterminedType), elements_(objs.begin(), objs.end()) {}

  SparseTensorType(const TypeId object_type, const TypePtrList &objs)
      : Object(object_type, kObjectTypeUndeterminedType), elements_(objs.begin(), objs.end()) {}

  /// \brief Destructor of SparseTensorType.
  ~SparseTensorType() override = default;
  MS_DECLARE_PARENT(SparseTensorType, Object)

  enum StringType : int { kToString = 0, kDumpText, kReprString };

  virtual std::string GetSparseTensorTypeName() const { return "SparseTensorType"; }
  virtual size_t GetElementIndex() { return 0; }
  virtual TypePtr element_type() {
    if (elements_.empty()) {
      return nullptr;
    }
    return elements_[GetElementIndex()];
  }
  std::string ElementsDtypeStr(const StringType str_type) const;
  TypeId generic_type_id() const override { return kObjectTypeSparseTensorType; }

  const TypePtr operator[](std::size_t dim) const;
  bool operator==(const Type &other) const override;
  size_t hash() const override;
  TypePtrList elements() const { return elements_; }

  std::size_t size() const { return elements_.size(); }
  std::string ToString() const override;
  std::string ToReprString() const override;
  std::string DumpText() const override;
  const TypePtrList ElementsClone() const;
  TypePtr DeepCopy() const override;

 private:
  TypePtrList elements_;
};
using SparseTensorTypePtr = std::shared_ptr<SparseTensorType>;

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
  size_t hash() const override;

 private:
  TypePtr element_type_;
};
using RowTensorTypePtr = std::shared_ptr<RowTensorType>;

/// \brief COOTensorType defines interface for coo tensor data type.
class MS_CORE_API COOTensorType final : public SparseTensorType {
 public:
  /// \brief Default constructor for COOTensorType.
  COOTensorType() : SparseTensorType(kObjectTypeCOOTensorType) {}

  /// \brief Constructor for COOTensorType.
  ///
  /// \param[in] obj The list of COOTensorType.
  explicit COOTensorType(const TypePtrList &obj) : SparseTensorType(kObjectTypeCOOTensorType, obj) {}

  /// \brief Destructor of COOTensorType.
  ~COOTensorType() override = default;
  MS_DECLARE_PARENT(COOTensorType, SparseTensorType)

  std::string GetSparseTensorTypeName() const override { return "COOTensor"; }
  size_t GetElementIndex() override { return 1; }

  TypeId generic_type_id() const override { return kObjectTypeCOOTensorType; }
  TypePtr DeepCopy() const override;
};
using COOTensorTypePtr = std::shared_ptr<COOTensorType>;

/// \brief CSRTensorType defines interface for csr tensor data type.
class MS_CORE_API CSRTensorType : public SparseTensorType {
 public:
  /// \brief Default constructor for CSRTensorType.
  CSRTensorType() : SparseTensorType(kObjectTypeCSRTensorType) {}

  /// \brief Constructor for CSRTensorType.
  ///
  /// \param[in] obj The list of CSRTensorType.
  explicit CSRTensorType(const TypePtrList &obj) : SparseTensorType(kObjectTypeCSRTensorType, obj) {}

  /// \brief Destructor of CSRTensorType.
  ~CSRTensorType() override = default;
  MS_DECLARE_PARENT(CSRTensorType, SparseTensorType)

  std::string GetSparseTensorTypeName() const override { return "CSRTensor"; }
  size_t GetElementIndex() override { return 2; }
  TypeId generic_type_id() const override { return kObjectTypeCSRTensorType; }
  TypePtr DeepCopy() const override;
};
using CSRTensorTypePtr = std::shared_ptr<CSRTensorType>;

/// \brief MapTensorType defines interface for map tensor data type.
class MS_CORE_API MapTensorType final : public Object {
 public:
  /// \brief Construct a generic MapTensorType.
  MapTensorType() : Object(kObjectTypeMapTensorType, true) {}

  /// \brief Construct a MapTensorType.
  ///
  /// \param[in] key The key data type.
  /// \param[in] value The value data type.
  explicit MapTensorType(const TypePtr &key, const TypePtr &value)
      : Object(kObjectTypeMapTensorType, false), key_dtype_(key), value_dtype_(value) {}

  /// \brief Destructor of MapTensorType.
  ~MapTensorType() override = default;
  MS_DECLARE_PARENT(MapTensorType, Object)

  TypeId generic_type_id() const override { return kObjectTypeMapTensorType; }

  /// \brief Get the key data type of this MapTensorType.
  ///
  /// \return The key data type.
  const TypePtr &key_dtype() const { return key_dtype_; }

  /// \brief Get the value data type of this MapTensorType.
  ///
  /// \return The key data type.
  const TypePtr &value_dtype() const { return value_dtype_; }

  TypePtr DeepCopy() const override;
  std::string ToString() const override;
  std::string ToReprString() const override;
  std::string DumpText() const override;
  bool operator==(const Type &other) const override;
  size_t hash() const override;

 private:
  TypePtr key_dtype_;
  TypePtr value_dtype_;
};
using MapTensorTypePtr = std::shared_ptr<MapTensorType>;
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_DTYPE_TENSORTYPE_H_
