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

#ifndef MINDSPORE_CCSRC_IR_DTYPE_CONTAINER_H_
#define MINDSPORE_CCSRC_IR_DTYPE_CONTAINER_H_

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
#include "ir/base.h"
#include "ir/named.h"
#include "ir/dtype/type.h"

namespace mindspore {
// TypeRefKey type

// List
class List : public Object {
 public:
  List() : Object(kObjectTypeList) {}
  List(const std::initializer_list<TypePtr> &objs)
      : Object(kObjectTypeList, false), elements_(objs.begin(), objs.end()) {}
  // Shadow copy;
  explicit List(const TypePtrList &obj) : Object(kObjectTypeList, false), elements_(obj) {}
  ~List() override {}
  MS_DECLARE_PARENT(List, Object)

  const TypePtr operator[](size_t dim) const;
  TypeId generic_type_id() const override { return kObjectTypeList; }
  TypePtr DeepCopy() const override;

  bool operator==(const Type &other) const override;
  std::size_t size() const { return elements_.size(); }
  TypePtrList elements() const { return elements_; }
  std::string ToString() const override;
  std::string ToReprString() const override { return "list_"; }
  std::string DumpText() const override;

 private:
  TypePtrList elements_;
};
using ListPtr = std::shared_ptr<List>;

using ClassAttrVector = std::vector<std::pair<std::string, TypePtr>>;

class Class : public Object {
 public:
  Class() : Object(kObjectTypeClass), tag_(Named("Class")) {}
  Class(const Named &tag, const ClassAttrVector &attributes, const std::unordered_map<std::string, ValuePtr> &methods);
  ~Class() override {}
  MS_DECLARE_PARENT(Class, Object)

  TypeId generic_type_id() const override { return kObjectTypeClass; }

  bool operator==(const Type &other) const override;
  TypePtr DeepCopy() const override;
  std::string ToString() const override;
  std::string DumpText() const override;
  void set_value(const std::unordered_map<std::string, ValuePtr> &v) { attributes_value_ = v; }

  Named tag() { return tag_; }
  std::unordered_map<std::string, ValuePtr> GetValue() { return attributes_value_; }
  std::unordered_map<std::string, ValuePtr> methods() { return methods_; }
  ClassAttrVector &GetAttributes() { return attributes_; }

  ClassAttrVector attributes_;

 private:
  Named tag_;
  std::unordered_map<std::string, ValuePtr> methods_;
  // For AbstractClass build value
  std::unordered_map<std::string, ValuePtr> attributes_value_;
};
using ClassPtr = std::shared_ptr<Class>;

class Tuple : public Object {
 public:
  Tuple() : Object(kObjectTypeTuple) {}
  // usage : Tuple t = {std::make_shared<Bool>(), std::make_shared<Int>(32)};
  Tuple(const std::initializer_list<TypePtr> &objs)
      : Object(kObjectTypeTuple, false), elements_(objs.begin(), objs.end()) {}

  // Shadow copy
  explicit Tuple(const TypePtrList &objs) : Object(kObjectTypeTuple, false), elements_(objs.begin(), objs.end()) {}

  ~Tuple() override {}
  MS_DECLARE_PARENT(Tuple, Object)

  TypeId generic_type_id() const override { return kObjectTypeTuple; }
  TypePtr DeepCopy() const override;

  std::string ToString() const override;
  std::string ToReprString() const override { return "tuple_"; }
  std::string DumpText() const override;
  const TypePtr operator[](size_t dim) const;
  bool operator==(const Type &other) const override;

  TypePtrList elements() const { return elements_; }
  std::size_t size() const { return elements_.size(); }

 private:
  TypePtrList elements_;
};
using TuplePtr = std::shared_ptr<Tuple>;

class Dictionary : public Object {
 public:
  Dictionary() : Object(kObjectTypeDictionary) {}
  explicit Dictionary(const std::vector<std::pair<std::string, TypePtr>> &key_values)
      : Object(kObjectTypeDictionary, false), key_values_(key_values) {}

  ~Dictionary() override {}
  MS_DECLARE_PARENT(Dictionary, Object)

  TypeId generic_type_id() const override { return kObjectTypeDictionary; }

  bool operator==(const Type &other) const override;
  TypePtr DeepCopy() const override;
  std::string ToString() const override;
  std::string DumpText() const override;

 private:
  std::vector<std::pair<std::string, TypePtr>> key_values_;
};
using DictionaryPtr = std::shared_ptr<Dictionary>;
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_IR_DTYPE_CONTAINER_H_
