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

#include "ir/value.h"

#include <algorithm>
#include <memory>
#include <cmath>
#include <cfloat>

#include "utils/convert_utils_base.h"

namespace mindspore {
const ValuePtr ValueSequence::operator[](const std::size_t &dim) const {
  if (dim >= size()) {
    MS_LOG(EXCEPTION) << "List index [" << dim << "] is out of range [" << size() << "].";
  }
  return elements_[dim];
}

bool ValueSequence::erase(size_t idx) {
  if (idx < size()) {
    (void)elements_.erase(elements_.begin() + SizeToInt(idx));
    return true;
  } else {
    return false;
  }
}

bool BoolImm::operator==(const Value &other) const {
  if (other.isa<BoolImm>()) {
    auto other_ = static_cast<const BoolImm &>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool BoolImm::operator==(const BoolImm &other) const { return v_ == other.v_; }

bool Int8Imm::operator==(const Value &other) const {
  if (other.isa<Int8Imm>()) {
    auto other_ = static_cast<const Int8Imm &>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool Int8Imm::operator==(const Int8Imm &other) const { return v_ == other.v_; }
bool Int16Imm::operator==(const Value &other) const {
  if (other.isa<Int16Imm>()) {
    auto other_ = static_cast<const Int16Imm &>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool Int16Imm::operator==(const Int16Imm &other) const { return v_ == other.v_; }
bool Int32Imm::operator==(const Value &other) const {
  if (other.isa<Int32Imm>()) {
    auto other_ = static_cast<const Int32Imm &>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool Int32Imm::operator==(const Int32Imm &other) const { return v_ == other.v_; }
bool Int64Imm::operator==(const Value &other) const {
  if (other.isa<Int64Imm>()) {
    auto other_ = static_cast<const Int64Imm &>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool Int64Imm::operator==(const Int64Imm &other) const { return v_ == other.v_; }
bool UInt8Imm::operator==(const Value &other) const {
  if (other.isa<UInt8Imm>()) {
    auto other_ = static_cast<const UInt8Imm &>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool UInt8Imm::operator==(const UInt8Imm &other) const { return v_ == other.v_; }
bool UInt16Imm::operator==(const Value &other) const {
  if (other.isa<UInt16Imm>()) {
    auto other_ = static_cast<const UInt16Imm &>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool UInt16Imm::operator==(const UInt16Imm &other) const { return v_ == other.v_; }
bool UInt32Imm::operator==(const Value &other) const {
  if (other.isa<UInt32Imm>()) {
    auto other_ = static_cast<const UInt32Imm &>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool UInt32Imm::operator==(const UInt32Imm &other) const { return v_ == other.v_; }
bool UInt64Imm::operator==(const Value &other) const {
  if (other.isa<UInt64Imm>()) {
    auto other_ = static_cast<const UInt64Imm &>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool UInt64Imm::operator==(const UInt64Imm &other) const { return v_ == other.v_; }

bool FP32Imm::operator==(const Value &other) const {
  if (other.isa<FP32Imm>()) {
    auto other_ = static_cast<const FP32Imm &>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool FP32Imm::operator==(const FP32Imm &other) const {
  if ((std::isinf(v_) && std::isinf(other.v_)) || (std::isnan(v_) && std::isnan(other.v_))) {
    return true;
  }
  return fabs(v_ - other.v_) < DBL_EPSILON;
}
bool FP64Imm::operator==(const Value &other) const {
  if (other.isa<FP64Imm>()) {
    auto other_ = static_cast<const FP64Imm &>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool ValueSequence::operator==(const Value &other) const {
  if (other.isa<ValueSequence>()) {
    auto other_ = static_cast<const ValueSequence &>(other);
    return *this == other_;
  } else {
    return false;
  }
}
bool ValueSequence::operator==(const ValueSequence &other) const {
  if (other.elements_.size() != elements_.size()) {
    return false;
  }
  return std::equal(elements_.begin(), elements_.end(), other.elements_.begin(),
                    [](const ValuePtr &lhs, const ValuePtr &rhs) { return *lhs == *rhs; });
}

std::string ValueSequence::ToString() const {
  std::ostringstream buffer;
  bool begin = true;
  for (auto &attr : elements_) {
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

std::string ValueSequence::DumpText() const {
  std::ostringstream oss;
  for (size_t i = 0; i < elements_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(elements_[i]);
    oss << (i > 0 ? ", " : "") << elements_[i]->DumpText();
  }
  return oss.str();
}

bool FP64Imm::operator==(const FP64Imm &other) const {
  if ((std::isinf(v_) && std::isinf(other.v_)) || (std::isnan(v_) && std::isnan(other.v_))) {
    return true;
  }
  return fabs(v_ - other.v_) < DBL_EPSILON;
}
bool StringImm::operator==(const Value &other) const {
  if (other.isa<StringImm>()) {
    auto other_ = static_cast<const StringImm &>(other);
    return *this == other_;
  }
  return false;
}

bool StringImm::operator==(const StringImm &other) const { return str_ == other.str_; }

bool AnyValue::operator==(const Value &other) const { return other.isa<AnyValue>(); }

bool ErrorValue::operator==(const Value &other) const {
  if (other.isa<ErrorValue>()) {
    auto other_ = static_cast<const ErrorValue &>(other);
    return err_type_ == other_.err_type_;
  }
  return false;
}

bool ErrorValue::operator==(const ErrorValue &other) const { return err_type_ == other.err_type_; }

std::size_t ValueSlice::hash() const {
  MS_EXCEPTION_IF_NULL(start_);
  MS_EXCEPTION_IF_NULL(stop_);
  MS_EXCEPTION_IF_NULL(step_);
  return hash_combine({tid(), start_->hash(), stop_->hash(), step_->hash()});
}

bool ValueSlice::operator==(const Value &other) const {
  if (other.isa<ValueSlice>()) {
    auto other_ = static_cast<const ValueSlice &>(other);
    return *this == other_;
  } else {
    return false;
  }
}

bool ValueSlice::operator==(const ValueSlice &other) const {
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

std::size_t KeywordArg::hash() const {
  MS_EXCEPTION_IF_NULL(value_);
  return hash_combine({tid(), std::hash<std::string>{}(key_), value_->hash()});
}

bool KeywordArg::operator==(const Value &other) const {
  if (other.isa<KeywordArg>()) {
    auto other_ = static_cast<const KeywordArg &>(other);
    return *this == other_;
  } else {
    return false;
  }
}

bool KeywordArg::operator==(const KeywordArg &other) const { return (other.key_ == key_ && *other.value_ == *value_); }

std::string KeywordArg::ToString() const {
  std::ostringstream buffer;
  buffer << "KeywordArg[";
  buffer << "key : " << key_;
  MS_EXCEPTION_IF_NULL(value_);
  buffer << ", value : " << value_->ToString();
  buffer << "]";
  return buffer.str();
}

const ValuePtr ValueDictionary::operator[](const ValuePtr &key) const {
  auto it = std::find_if(key_values_.cbegin(), key_values_.cend(),
                         [key](const std::pair<ValuePtr, ValuePtr> &item) { return item.first == key; });
  if (it == key_values_.end()) {
    MS_LOG(EXCEPTION) << "The key " << key->ToString() << " is not in the map";
  }
  return it->second;
}

bool ValueDictionary::operator==(const Value &other) const {
  if (other.isa<ValueDictionary>()) {
    auto other_ = static_cast<const ValueDictionary &>(other);
    return *this == other_;
  } else {
    return false;
  }
}

bool ValueDictionary::operator==(const ValueDictionary &other) const {
  if (key_values_.size() != other.key_values_.size()) {
    return false;
  }
  for (size_t index = 0; index < key_values_.size(); index++) {
    if (!(*key_values_[index].first == *other.key_values_[index].first) ||
        !(*key_values_[index].second == *other.key_values_[index].second))
      return false;
  }
  return true;
}

bool UMonad::operator==(const Value &other) const { return other.isa<UMonad>(); }
const ValuePtr kUMonad = std::make_shared<UMonad>();

bool IOMonad::operator==(const Value &other) const { return other.isa<IOMonad>(); }
const ValuePtr kIOMonad = std::make_shared<IOMonad>();
}  // namespace mindspore
