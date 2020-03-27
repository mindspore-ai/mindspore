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

#include "ir/value.h"
#include <algorithm>
#include <memory>
#include <cmath>
#include <cfloat>

#include "pybind_api/api_register.h"
#include "pipeline/static_analysis/abstract_value.h"

namespace mindspore {
const ValuePtr ValueSequeue::operator[](const std::size_t& dim) const {
  if (dim >= size()) {
    MS_LOG(EXCEPTION) << "List index [" << dim << "] is out of range [" << size() << "].";
  }
  return elements_[dim];
}

bool ValueSequeue::erase(size_t idx) {
  if (idx < size()) {
    (void)elements_.erase(elements_.begin() + SizeToInt(idx));
    return true;
  } else {
    return false;
  }
}

bool BoolImm::operator==(const Value& other) const {
  if (other.isa<BoolImm>()) {
    auto other_ = static_cast<const BoolImm&>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool BoolImm::operator==(const BoolImm& other) const { return v_ == other.v_; }

bool Int8Imm::operator==(const Value& other) const {
  if (other.isa<Int8Imm>()) {
    auto other_ = static_cast<const Int8Imm&>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool Int8Imm::operator==(const Int8Imm& other) const { return v_ == other.v_; }
bool Int16Imm::operator==(const Value& other) const {
  if (other.isa<Int16Imm>()) {
    auto other_ = static_cast<const Int16Imm&>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool Int16Imm::operator==(const Int16Imm& other) const { return v_ == other.v_; }
bool Int32Imm::operator==(const Value& other) const {
  if (other.isa<Int32Imm>()) {
    auto other_ = static_cast<const Int32Imm&>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool Int32Imm::operator==(const Int32Imm& other) const { return v_ == other.v_; }
bool Int64Imm::operator==(const Value& other) const {
  if (other.isa<Int64Imm>()) {
    auto other_ = static_cast<const Int64Imm&>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool Int64Imm::operator==(const Int64Imm& other) const { return v_ == other.v_; }
bool UInt8Imm::operator==(const Value& other) const {
  if (other.isa<UInt8Imm>()) {
    auto other_ = static_cast<const UInt8Imm&>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool UInt8Imm::operator==(const UInt8Imm& other) const { return v_ == other.v_; }
bool UInt16Imm::operator==(const Value& other) const {
  if (other.isa<UInt16Imm>()) {
    auto other_ = static_cast<const UInt16Imm&>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool UInt16Imm::operator==(const UInt16Imm& other) const { return v_ == other.v_; }
bool UInt32Imm::operator==(const Value& other) const {
  if (other.isa<UInt32Imm>()) {
    auto other_ = static_cast<const UInt32Imm&>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool UInt32Imm::operator==(const UInt32Imm& other) const { return v_ == other.v_; }
bool UInt64Imm::operator==(const Value& other) const {
  if (other.isa<UInt64Imm>()) {
    auto other_ = static_cast<const UInt64Imm&>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool UInt64Imm::operator==(const UInt64Imm& other) const { return v_ == other.v_; }
bool FP32Imm::operator==(const Value& other) const {
  if (other.isa<FP32Imm>()) {
    auto other_ = static_cast<const FP32Imm&>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool FP32Imm::operator==(const FP32Imm& other) const { return fabs(v_ - other.v_) < FLT_EPSILON; }
bool FP64Imm::operator==(const Value& other) const {
  if (other.isa<FP64Imm>()) {
    auto other_ = static_cast<const FP64Imm&>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool ValueSequeue::operator==(const Value& other) const {
  if (other.isa<ValueSequeue>()) {
    auto other_ = static_cast<const ValueSequeue&>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool ValueSequeue::operator==(const ValueSequeue& other) const {
  if (other.elements_.size() != elements_.size()) {
    return false;
  }
  return std::equal(elements_.begin(), elements_.end(), other.elements_.begin(),
                    [](const ValuePtr& lhs, const ValuePtr& rhs) { return *lhs == *rhs; });
}

std::string ValueSequeue::ToString() const {
  std::ostringstream buffer;
  bool begin = true;
  for (auto& attr : elements_) {
    if (!begin) {
      buffer << ", ";
    } else {
      begin = false;
    }
    MS_EXCEPTION_IF_NULL(attr);
    buffer << attr->ToString();
  }
  return buffer.str();
}

std::string ValueSequeue::DumpText() const {
  std::ostringstream oss;
  for (size_t i = 0; i < elements_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(elements_[i]);
    oss << (i > 0 ? ", " : "") << elements_[i]->DumpText();
  }
  return oss.str();
}

bool FP64Imm::operator==(const FP64Imm& other) const { return fabs(v_ - other.v_) < DBL_EPSILON; }
bool StringImm::operator==(const Value& other) const {
  if (other.isa<StringImm>()) {
    auto other_ = static_cast<const StringImm&>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool StringImm::operator==(const StringImm& other) const { return str_ == other.str_; }

bool RefKey::operator==(const Value& other) const {
  if (other.isa<RefKey>()) {
    auto other_ = static_cast<const RefKey&>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool RefKey::operator==(const RefKey& other) const { return tag_ == other.tag_; }

bool AnyValue::operator==(const Value& other) const {
  if (other.isa<AnyValue>()) {
    return true;
  } else {
    return false;
  }
}
const ValuePtr kAnyValue = std::make_shared<AnyValue>();
using ContextPtr = abstract::AnalysisContextPtr;

abstract::AbstractBasePtr Scalar::ToAbstract() {
  return std::make_shared<abstract::AbstractScalar>(shared_from_base<Value>());
}

abstract::AbstractBasePtr StringImm::ToAbstract() {
  return std::make_shared<abstract::AbstractScalar>(shared_from_base<Value>(), std::make_shared<String>());
}

abstract::AbstractBasePtr RefKey::ToAbstract() {
  auto refkey = std::make_shared<abstract::AbstractRefKey>();
  refkey->set_value(shared_from_base<Value>());
  return refkey;
}

abstract::AbstractBasePtr AnyValue::ToAbstract() { return std::make_shared<abstract::AbstractScalar>(); }

abstract::AbstractBasePtr ValueTuple::ToAbstract() {
  abstract::AbstractBasePtrList a_list;
  (void)std::transform(elements_.begin(), elements_.end(), std::back_inserter(a_list), [](const ValuePtr& ele) {
    MS_EXCEPTION_IF_NULL(ele);
    return ele->ToAbstract();
  });
  return std::make_shared<abstract::AbstractTuple>(a_list);
}

abstract::AbstractBasePtr ValueList::ToAbstract() {
  abstract::AbstractBasePtrList a_list;
  (void)std::transform(elements_.begin(), elements_.end(), std::back_inserter(a_list), [](const ValuePtr& ele) {
    MS_EXCEPTION_IF_NULL(ele);
    return ele->ToAbstract();
  });
  return std::make_shared<abstract::AbstractList>(a_list);
}

std::size_t ValueSlice::hash() const {
  MS_EXCEPTION_IF_NULL(start_);
  MS_EXCEPTION_IF_NULL(stop_);
  MS_EXCEPTION_IF_NULL(step_);
  return hash_combine({tid(), start_->hash(), stop_->hash(), step_->hash()});
}

bool ValueSlice::operator==(const Value& other) const {
  if (other.isa<ValueSlice>()) {
    auto other_ = static_cast<const ValueSlice&>(other);
    return *this == other_;
  } else {
    return false;
  }
}

bool ValueSlice::operator==(const ValueSlice& other) const {
  MS_EXCEPTION_IF_NULL(start_);
  MS_EXCEPTION_IF_NULL(stop_);
  MS_EXCEPTION_IF_NULL(step_);
  return (*start_ == *other.start_ && *stop_ == *other.stop_ && *step_ == *other.step_);
}

std::string ValueSlice::ToString() const {
  MS_EXCEPTION_IF_NULL(start_);
  MS_EXCEPTION_IF_NULL(stop_);
  MS_EXCEPTION_IF_NULL(step_);
  std::ostringstream buffer;
  buffer << "Slice[";
  buffer << start_->ToString() << " : ";
  buffer << stop_->ToString() << " : ";
  buffer << step_->ToString();
  buffer << "]";
  return buffer.str();
}

abstract::AbstractBasePtr ValueSlice::ToAbstract() {
  MS_EXCEPTION_IF_NULL(start_);
  MS_EXCEPTION_IF_NULL(stop_);
  MS_EXCEPTION_IF_NULL(step_);
  abstract::AbstractBasePtr start = start_->ToAbstract();
  abstract::AbstractBasePtr end = stop_->ToAbstract();
  abstract::AbstractBasePtr step = step_->ToAbstract();
  return std::make_shared<abstract::AbstractSlice>(start, end, step);
}

std::size_t KeywordArg::hash() const {
  MS_EXCEPTION_IF_NULL(value_);
  return hash_combine({tid(), std::hash<std::string>{}(key_), value_->hash()});
}

bool KeywordArg::operator==(const Value& other) const {
  if (other.isa<KeywordArg>()) {
    auto other_ = static_cast<const KeywordArg&>(other);
    return *this == other_;
  } else {
    return false;
  }
}

bool KeywordArg::operator==(const KeywordArg& other) const { return (other.key_ == key_ && *other.value_ == *value_); }

std::string KeywordArg::ToString() const {
  std::ostringstream buffer;
  buffer << "KeywordArg[";
  buffer << "key : " << key_;
  MS_EXCEPTION_IF_NULL(value_);
  buffer << "value : " << value_->ToString();
  buffer << "]";
  return buffer.str();
}

abstract::AbstractBasePtr KeywordArg::ToAbstract() {
  MS_EXCEPTION_IF_NULL(value_);
  abstract::AbstractBasePtr argument = value_->ToAbstract();
  return std::make_shared<abstract::AbstractKeywordArg>(key_, argument);
}

const ValuePtr ValueDictionary::operator[](const std::string& key) const {
  auto it = std::find_if(key_values_.begin(), key_values_.end(),
                         [key](const std::pair<std::string, ValuePtr>& item) { return item.first == key; });
  if (it == key_values_.end()) {
    MS_LOG(EXCEPTION) << "The key " << key << " is not in the map";
  }
  return it->second;
}

bool ValueDictionary::operator==(const Value& other) const {
  if (other.isa<ValueDictionary>()) {
    auto other_ = static_cast<const ValueDictionary&>(other);
    return *this == other_;
  } else {
    return false;
  }
}

bool ValueDictionary::operator==(const ValueDictionary& other) const {
  if (key_values_.size() != other.key_values_.size()) {
    return false;
  }
  for (size_t index = 0; index < key_values_.size(); index++) {
    if (key_values_[index].first != other.key_values_[index].first) {
      return false;
    }
    if (!(*key_values_[index].second == *other.key_values_[index].second)) {
      return false;
    }
  }
  return true;
}

abstract::AbstractBasePtr ValueDictionary::ToAbstract() {
  std::vector<std::pair<std::string, abstract::AbstractBasePtr>> kv;
  (void)std::transform(
    key_values_.begin(), key_values_.end(), std::back_inserter(kv),
    [](const std::pair<std::string, ValuePtr>& item) { return std::make_pair(item.first, item.second->ToAbstract()); });
  return std::make_shared<abstract::AbstractDictionary>(kv);
}

REGISTER_PYBIND_DEFINE(
  RefKey, ([](const py::module* m) {
    (void)py::class_<RefKey, std::shared_ptr<RefKey>>(*m, "RefKey").def(py::init<std::string>(), py::arg("tag"));
  }));
}  // namespace mindspore
