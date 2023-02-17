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

#include "ir/dtype/container.h"
#include <string>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <type_traits>
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"

namespace mindspore {
static std::string DumpTypeVector(const std::vector<TypePtr> &elements, bool is_dumptext, bool is_dynamic = false,
                                  const TypePtr &dynamic_element_type = nullptr) {
  std::ostringstream oss;
  if (is_dynamic) {
    oss << "element: ";
    if (dynamic_element_type != nullptr) {
      oss << (is_dumptext ? dynamic_element_type->DumpText() : dynamic_element_type->ToString());
    } else {
      oss << "Undetermined";
    }
    return oss.str();
  }
  if (elements.empty() || elements.front().get() == nullptr) {
    return oss.str();
  }
  bool begin = true;
  size_t cnt = 0;
  // write 'Tuple[Bool, Bool, Bool, Int, Float, Float]' as 'Tuple[Bool...3, Int, Float...2]'
  for (size_t i = 0; i < elements.size(); ++i) {
    TypePtr elem = elements[i];
    cnt += 1;
    bool print = false;
    if (i + 1 < elements.size()) {
      TypePtr next = elements[i + 1];
      if (*elem != *next) {
        print = true;
      }
    } else {
      // encounter last element
      print = true;
    }
    if (!print) {
      continue;
    }
    if (!begin) {
      oss << ",";
    } else {
      begin = false;
    }
    oss << (is_dumptext ? elem->DumpText() : elem->ToString());
    if (cnt > 1) {
      oss << "*" << cnt;
    }
    cnt = 0;
  }
  return oss.str();
}

TypePtr List::DeepCopy() const {
  ListPtr ret = nullptr;
  if (IsGeneric()) {
    ret = std::make_shared<List>();
  } else {
    TypePtrList elements;
    (void)std::transform(elements_.begin(), elements_.end(), std::back_inserter(elements),
                         [](const TypePtr &ele) { return ele->DeepCopy(); });
    ret = std::make_shared<List>(elements);
  }
  if (dynamic_len_) {
    ret->set_dynamic_len(true);
    ret->set_dynamic_element_type(dynamic_element_type_);
  }
  return ret;
}

const TypePtr List::operator[](std::size_t dim) const {
  if (dynamic_len_) {
    MS_LOG(EXCEPTION) << "Dynamic length list " << ToString() << " can not get element.";
  }
  if (dim >= size()) {
    MS_LOG(EXCEPTION) << "Index " << dim << " is out range of the List size " << size() << ".";
  }
  return elements_[dim];
}

bool List::operator==(const Type &other) const {
  if (!IsSameObjectType(*this, other)) {
    return false;
  }
  const List &other_list = static_cast<const List &>(other);
  if (dynamic_len_ || other_list.dynamic_len()) {
    if (dynamic_len_ != other_list.dynamic_len()) {
      return false;
    }
    if (dynamic_element_type_ == other_list.dynamic_element_type()) {
      return true;
    }
    if (dynamic_element_type_ == nullptr || other_list.dynamic_element_type() == nullptr) {
      return false;
    }
    return *dynamic_element_type_ == *(other_list.dynamic_element_type());
  }
  return TypeListEqual()(elements_, other_list.elements_);
}

size_t List::hash() const {
  size_t hash_value = hash_combine(static_cast<size_t>(kMetaTypeObject), static_cast<size_t>(object_type()));
  if (dynamic_len_) {
    size_t next_hash_value = hash_combine(hash_value, static_cast<size_t>(dynamic_len_));
    if (dynamic_element_type_ != nullptr) {
      return hash_combine(next_hash_value, static_cast<size_t>(dynamic_element_type_->object_type()));
    }
    return next_hash_value;
  }
  return hash_combine(hash_value, TypeListHasher()(elements_));
}

std::string List::DumpContent(bool is_dumptext) const {
  std::ostringstream buffer;
  auto type_name = dynamic_len_ ? "Dynamic List" : "List";
  if (IsGeneric()) {
    buffer << type_name;
  } else {
    buffer << type_name;
    buffer << "[";
    buffer << DumpTypeVector(elements_, is_dumptext, dynamic_len_, dynamic_element_type_);
    buffer << "]";
  }
  return buffer.str();
}

TypePtr List::dynamic_element_type() const {
  if (!dynamic_len_) {
    MS_LOG(EXCEPTION) << "Constant list " << ToString() << " can not get the dynamic element type.";
  }
  return dynamic_element_type_;
}

void List::set_dynamic_element_type(TypePtr dynamic_element_type) {
  if (!dynamic_len_) {
    MS_LOG(EXCEPTION) << "Constant list " << ToString() << " can not set the dynamic element type.";
  }
  dynamic_element_type_ = dynamic_element_type;
}

TypePtr Tuple::DeepCopy() const {
  TuplePtr ret = nullptr;
  if (IsGeneric()) {
    ret = std::make_shared<Tuple>();
  } else {
    TypePtrList elements;
    (void)std::transform(elements_.begin(), elements_.end(), std::back_inserter(elements),
                         [](const TypePtr &ele) { return ele->DeepCopy(); });
    ret = std::make_shared<Tuple>(elements);
  }
  if (dynamic_len_) {
    ret->set_dynamic_len(true);
    ret->set_dynamic_element_type(dynamic_element_type_);
  }
  return ret;
}

bool Tuple::operator==(const Type &other) const {
  if (!IsSameObjectType(*this, other)) {
    return false;
  }
  auto other_tuple = static_cast<const Tuple &>(other);
  if (dynamic_len_ || other_tuple.dynamic_len()) {
    if (dynamic_len_ != other_tuple.dynamic_len()) {
      return false;
    }
    if (dynamic_element_type_ == other_tuple.dynamic_element_type()) {
      return true;
    }
    if (dynamic_element_type_ == nullptr || other_tuple.dynamic_element_type() == nullptr) {
      return false;
    }
    return *dynamic_element_type_ == *(other_tuple.dynamic_element_type());
  }
  return TypeListEqual()(elements_, other_tuple.elements_);
}

size_t Tuple::hash() const {
  size_t hash_value = hash_combine(static_cast<size_t>(kMetaTypeObject), static_cast<size_t>(object_type()));
  if (dynamic_len_) {
    size_t next_hash_value = hash_combine(hash_value, static_cast<size_t>(dynamic_len_));
    if (dynamic_element_type_ != nullptr) {
      return hash_combine(next_hash_value, static_cast<size_t>(dynamic_element_type_->object_type()));
    }
    return next_hash_value;
  }
  return hash_combine(hash_value, TypeListHasher()(elements_));
}

const TypePtr Tuple::operator[](std::size_t dim) const {
  if (dynamic_len_) {
    MS_LOG(EXCEPTION) << "Dynamic length tuple " << ToString() << " can not get element.";
  }
  if (dim >= size()) {
    MS_LOG(EXCEPTION) << "Index " << dim << " is out range of the Tuple size " << size() << ".";
  }
  return elements_[dim];
}

std::string Tuple::DumpContent(bool is_dumptext) const {
  std::ostringstream buffer;
  auto type_name = dynamic_len_ ? "Dynamic Tuple" : "Tuple";
  if (IsGeneric()) {
    buffer << type_name;
  } else {
    buffer << type_name;
    buffer << "[";
    buffer << DumpTypeVector(elements_, is_dumptext, dynamic_len_, dynamic_element_type_);
    buffer << "]";
  }
  return buffer.str();
}

TypePtr Tuple::dynamic_element_type() const {
  if (!dynamic_len_) {
    MS_LOG(EXCEPTION) << "Constant tuple " << ToString() << " can not get the dynamic element type.";
  }
  return dynamic_element_type_;
}

void Tuple::set_dynamic_element_type(TypePtr dynamic_element_type) {
  if (!dynamic_len_) {
    MS_LOG(EXCEPTION) << "Constant tuple " << ToString() << " can not set the dynamic element type.";
  }
  dynamic_element_type_ = dynamic_element_type;
}

TypePtr Dictionary::DeepCopy() const {
  if (IsGeneric()) {
    return std::make_shared<Dictionary>();
  } else {
    std::vector<std::pair<ValuePtr, TypePtr>> kv;
    (void)std::transform(
      key_values_.cbegin(), key_values_.cend(), std::back_inserter(kv),
      [](const std::pair<ValuePtr, TypePtr> &item) { return std::make_pair(item.first, item.second->DeepCopy()); });
    return std::make_shared<Dictionary>(kv);
  }
}

std::string DumpKeyVector(const std::vector<ValuePtr> &keys) {
  std::ostringstream buffer;
  for (const auto &key : keys) {
    buffer << key->ToString() << ",";
  }
  return buffer.str();
}

std::string Dictionary::DumpContent(bool) const {
  std::ostringstream buffer;
  std::vector<ValuePtr> keys;
  std::vector<TypePtr> values;
  for (const auto &kv : key_values_) {
    keys.push_back(kv.first);
    values.push_back(kv.second);
  }
  if (IsGeneric()) {
    buffer << "Dictionary";
  } else {
    buffer << "Dictionary[";
    buffer << "[" << DumpKeyVector(keys) << "],";
    buffer << "[" << DumpTypeVector(values, false) << "]";
    buffer << "]";
  }
  return buffer.str();
}

bool Dictionary::operator==(const mindspore::Type &other) const {
  if (!IsSameObjectType(*this, other)) {
    return false;
  }

  const auto &other_dict = static_cast<const Dictionary &>(other);
  const auto size = key_values_.size();
  if (size != other_dict.key_values_.size()) {
    return false;
  }
  for (size_t i = 0; i < size; ++i) {
    const auto &a = key_values_[i];
    const auto &b = other_dict.key_values_[i];
    if (!common::IsEqual(a.first, b.first) || !common::IsEqual(a.second, b.second)) {
      return false;
    }
  }
  return true;
}

size_t Dictionary::hash() const {
  size_t hash_value = hash_combine(static_cast<size_t>(kMetaTypeObject), static_cast<size_t>(object_type()));
  hash_value = hash_combine(hash_value, key_values_.size());
  for (auto &kv : key_values_) {
    hash_value = hash_combine(hash_value, (kv.first == nullptr ? 0 : kv.first->hash()));
    hash_value = hash_combine(hash_value, (kv.second == nullptr ? 0 : kv.second->hash()));
  }
  return hash_value;
}
}  // namespace mindspore
