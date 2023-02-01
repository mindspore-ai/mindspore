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

#include "ir/dtype.h"
#include <cstdlib>
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"

namespace mindspore {
TypePtr Keyword::DeepCopy() const {
  if (IsGeneric()) {
    return std::make_shared<Keyword>();
  } else {
    MS_EXCEPTION_IF_NULL(value_);
    std::string key = key_;
    return std::make_shared<Keyword>(key, value_->DeepCopy());
  }
}

std::string Keyword::ToString() const {
  std::ostringstream buffer;
  if (IsGeneric()) {
    buffer << "Keyword";
  } else {
    MS_EXCEPTION_IF_NULL(value_);
    buffer << "Keyword[";
    buffer << "key : " << key_;
    buffer << ", value : " << value_->ToString();
    buffer << "]";
  }
  return buffer.str();
}

bool Keyword::operator==(const Type &other) const {
  if (!IsSameObjectType(*this, other)) {
    return false;
  }
  const auto &other_keyword = static_cast<const Keyword &>(other);
  return (other_keyword.key_ == key_ && *other_keyword.value_ == *value_);
}

std::string Keyword::DumpText() const { return ToString(); }

TypePtr Slice::DeepCopy() const {
  if (IsGeneric()) {
    return std::make_shared<Slice>();
  } else {
    MS_EXCEPTION_IF_NULL(start_);
    MS_EXCEPTION_IF_NULL(stop_);
    MS_EXCEPTION_IF_NULL(step_);
    auto copy = std::make_shared<Slice>(start_->DeepCopy(), stop_->DeepCopy(), step_->DeepCopy());
    return copy;
  }
}

std::string Slice::ToString() const {
  std::ostringstream buffer;
  if (IsGeneric()) {
    buffer << "Slice";
  } else {
    MS_EXCEPTION_IF_NULL(start_);
    MS_EXCEPTION_IF_NULL(stop_);
    MS_EXCEPTION_IF_NULL(step_);
    buffer << "Slice[";
    buffer << start_->ToString() << " : ";
    buffer << stop_->ToString() << " : ";
    buffer << step_->ToString();
    buffer << "]";
  }
  return buffer.str();
}

bool Slice::operator==(const Type &other) const {
  if (!IsSameObjectType(*this, other)) {
    return false;
  }
  auto other_slice = static_cast<const Slice &>(other);
  return common::IsEqual(start_, other_slice.start_) && common::IsEqual(stop_, other_slice.stop_) &&
         common::IsEqual(step_, other_slice.step_);
}

std::string Slice::DumpText() const { return ToString(); }

Function::Function() : Object(kObjectTypeFunction) {
  args_ = std::vector<TypePtr>();
  retval_ = nullptr;
}

Function::Function(const std::vector<TypePtr> &args, const TypePtr retval)
    : Object(kObjectTypeFunction, false), args_(args), retval_(retval) {}

TypePtr Function::DeepCopy() const {
  if (IsGeneric()) {
    return std::make_shared<Function>();
  } else {
    TypePtrList args;
    TypePtr retval = nullptr;
    (void)std::transform(args_.begin(), args_.end(), std::back_inserter(args),
                         [](const TypePtr &arg) { return arg->DeepCopy(); });
    if (retval_ != nullptr) {
      retval = retval_->DeepCopy();
    }
    return std::make_shared<Function>(args, retval);
  }
}

bool Function::operator==(const Type &other) const {
  if (!IsSameObjectType(*this, other)) {
    return false;
  }

  const auto &other_function = static_cast<const Function &>(other);
  if ((retval_ != nullptr) && (other_function.retval_ != nullptr)) {
    if (*retval_ != *other_function.retval_) {
      return false;
    }
  } else if ((retval_ == nullptr) && (other_function.retval_ != nullptr)) {
    return false;
  }
  if (args_.size() != other_function.args_.size()) {
    return false;
  }
  for (size_t i = 0; i < args_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(args_[i]);
    MS_EXCEPTION_IF_NULL(other_function.args_[i]);
    if (*args_[i] != *other_function.args_[i]) {
      return false;
    }
  }
  return true;
}

std::string Function::ToString() const {
  std::ostringstream buffer;
  if (IsGeneric()) {
    buffer << "Func";
  } else {
    buffer << "Func[(";
    bool begin = true;
    for (auto &attr : args_) {
      if (!begin) {
        buffer << ", ";
      } else {
        begin = false;
      }
      buffer << attr->ToString();
    }
    buffer << ")";
    if (retval_ != nullptr) {
      buffer << ", " << retval_->ToString() << "]";
    } else {
      buffer << "]";
    }
  }
  return buffer.str();
}

TypePtr JTagged::DeepCopy() const {
  MS_EXCEPTION_IF_NULL(subtype_);
  if (IsGeneric()) {
    return std::make_shared<JTagged>();
  } else {
    auto subtype = subtype_->DeepCopy();
    return std::make_shared<JTagged>(subtype);
  }
}

std::string JTagged::ToString() const {
  MS_EXCEPTION_IF_NULL(subtype_);
  std::ostringstream buffer;
  if (IsGeneric()) {
    buffer << "JT";
  } else {
    buffer << "JT[";
    buffer << subtype_->ToString() << "]";
  }
  return buffer.str();
}

std::string JTagged::DumpText() const {
  MS_EXCEPTION_IF_NULL(subtype_);
  std::ostringstream buffer;
  if (IsGeneric()) {
    buffer << "JT";
  } else {
    buffer << "JT[";
    buffer << subtype_->DumpText() << "]";
  }
  return buffer.str();
}

std::ostream &operator<<(std::ostream &os, const std::shared_ptr<Problem> problem) {
  MS_EXCEPTION_IF_NULL(problem);
  os << problem->ToString();
  return os;
}
}  // namespace mindspore
