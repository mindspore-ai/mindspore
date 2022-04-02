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

#include "ir/variable.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace {
void SetValueMutable(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractTensor>()) {
    return;
  }

  auto abs_sequence = abs->cast<abstract::AbstractSequencePtr>();
  if (abs_sequence != nullptr) {
    const auto &elements = abs_sequence->elements();
    for (auto &ele : elements) {
      SetValueMutable(ele);
    }
    return;
  }

  auto abs_dict = abs->cast<abstract::AbstractDictionaryPtr>();
  if (abs_dict != nullptr) {
    const auto &elements = abs_dict->elements();
    for (auto &ele : elements) {
      SetValueMutable(ele.second);
    }
    return;
  }

  abs->set_value_mutable(true);
}
}  // namespace

abstract::AbstractBasePtr Variable::ToAbstract() {
  if (real_value_ == nullptr) {
    MS_LOG(EXCEPTION) << "Get abstract failed. The real value of Variable has not been set.";
  }
  auto abs = real_value_->ToAbstract();
  SetValueMutable(abs);
  return abs;
}

bool Variable::operator==(const Variable &other) const {
  if (this == &other) {
    return true;
  }
  auto other_real_value = other.real_value();
  if (real_value_ == nullptr || other_real_value == nullptr) {
    return false;
  }
  return *real_value_ == *other_real_value;
}

std::string Variable::ToString() const {
  std::ostringstream oss;
  if (real_value_ == nullptr) {
    oss << "Variable(NULL)";
  } else {
    oss << "Variable(" << real_value_->ToString() << ")";
  }
  return oss.str();
}

std::string Variable::DumpText() const {
  std::ostringstream oss;
  if (real_value_ == nullptr) {
    oss << type_name() << "(NULL)";
  } else {
    oss << type_name() << "(" << real_value_->DumpText() << ")";
  }
  return oss.str();
}
}  // namespace mindspore
