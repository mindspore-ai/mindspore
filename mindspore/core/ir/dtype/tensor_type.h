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
#include <unordered_map>
#include <algorithm>
#include "base/base.h"
#include "ir/named.h"
#include "ir/dtype/type.h"

namespace mindspore {
class MS_CORE_API UndeterminedType : public Object {
 public:
  UndeterminedType() : Object(kObjectTypeUndeterminedType) {}
  explicit UndeterminedType(const TypePtr &ele)
      : Object(kObjectTypeUndeterminedType, kMetaTypeObject, false), element_type_(ele) {}
  ~UndeterminedType() override = default;
  MS_DECLARE_PARENT(UndeterminedType, Object)

  TypeId generic_type_id() const override { return kObjectTypeUndeterminedType; }
  const TypePtr element() const { return element_type_; }
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

class MS_CORE_API TensorType : public Object {
 public:
  TensorType() : Object(kObjectTypeTensorType, kObjectTypeUndeterminedType) {}
  explicit TensorType(const TypePtr &ele)
      : Object(kObjectTypeTensorType, kObjectTypeUndeterminedType, false), element_type_(ele) {}
  ~TensorType() override = default;
  MS_DECLARE_PARENT(TensorType, Object)

  TypeId generic_type_id() const override { return kObjectTypeTensorType; }
  const TypePtr element() const { return element_type_; }
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

class MS_CORE_API RowTensorType : public Object {
 public:
  RowTensorType() : Object(kObjectTypeRowTensorType, kObjectTypeUndeterminedType) {}
  explicit RowTensorType(const TypePtr &ele)
      : Object(kObjectTypeRowTensorType, kObjectTypeUndeterminedType, false), element_type_(ele) {}
  ~RowTensorType() override = default;
  MS_DECLARE_PARENT(RowTensorType, Object)

  TypeId generic_type_id() const override { return kObjectTypeRowTensorType; }
  const TypePtr element() const { return element_type_; }
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

class MS_CORE_API SparseTensorType : public Object {
 public:
  SparseTensorType() : Object(kObjectTypeSparseTensorType, kObjectTypeUndeterminedType) {}
  explicit SparseTensorType(const TypePtr &ele)
      : Object(kObjectTypeSparseTensorType, kObjectTypeUndeterminedType, false), element_type_(ele) {}
  ~SparseTensorType() override = default;
  MS_DECLARE_PARENT(SparseTensorType, Object)

  TypeId generic_type_id() const override { return kObjectTypeSparseTensorType; }
  const TypePtr element() const { return element_type_; }
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
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_DTYPE_TENSORTYPE_H_
