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

#include "base/base_ref.h"

namespace mindspore {
iterator ConstIteratorCast(std::vector<BaseRef> *v, const const_iterator iter) {
  return std::next(v->begin(), std::distance(v->cbegin(), iter));
}

BaseRef::BaseRef(const BaseRef &other) : Base(other), m_ptr(other.m_ptr) {
  if (!m_ptr) {
    m_ptr = other.copy();
  }
}

bool BaseRef::operator==(const BaseRef &other) const {
  if (m_ptr == other.m_ptr) {
    return true;
  }
  if (m_ptr == nullptr && other.m_ptr == nullptr) {
    return *this == other;
  }
  if (m_ptr == nullptr || other.m_ptr == nullptr) {
    return false;
  }
  if (type() != other.type()) {
    MS_LOG(DEBUG) << "Type mismatch";
    return false;
  }
  if (m_ptr->isa<Value>()) {
    return *(m_ptr->cast_ptr<Value>()) == *(other.m_ptr->cast_ptr<Value>());
  }

  // for noderef equal
  if (m_ptr->isa<BaseRef>()) {
    return *(m_ptr->cast_ptr<BaseRef>()) == *(other.m_ptr->cast_ptr<BaseRef>());
  }

  // for node equal
  return *m_ptr == *other.m_ptr;
}

// left reference
BaseRef &BaseRef::operator=(const BaseRef &other) {
  if ((m_ptr != nullptr && m_ptr == other.m_ptr) || this == &other) {
    return *this;
  }
  m_ptr = other.copy();
  return *this;
}

// right reference
BaseRef &BaseRef::operator=(BaseRef &&other) {
  if ((m_ptr != nullptr && m_ptr == other.m_ptr) || this == &other) {
    return *this;
  }
  m_ptr = other.copy();
  other.m_ptr = nullptr;
  return *this;
}

std::string BaseRef::ToString() const {
  if (m_ptr != nullptr) {
    return std::string(m_ptr->type_name()) + std::string(" value:") + m_ptr->ToString();
  }
  return std::string();
}

uint32_t BaseRef::type() const {
  if (m_ptr != nullptr) {
    return m_ptr->tid();
  }
  return tid();
}

// left reference
SetRef::SetRef(const SetRef &other) : elements_(other.elements_) {}

SetRef &SetRef::operator=(const SetRef &other) {
  if (elements_ == other.elements_ || this == &other) {
    return *this;
  }
  elements_ = other.elements_;
  return *this;
}

std::string SetRef::ToString() const {
  std::ostringstream buffer;
  bool begin = true;
  buffer << "set[";
  for (auto &attr : elements_) {
    if (!begin) {
      buffer << ", ";
    } else {
      begin = false;
    }
    buffer << attr.ToString();
  }
  buffer << "]";
  return buffer.str();
}

// left reference
VectorRef::VectorRef(const VectorRef &other) : elements_(other.elements_) {}

VectorRef &VectorRef::operator=(const VectorRef &other) {
  if (elements_ == other.elements_ || this == &other) {
    return *this;
  }
  elements_ = other.elements_;
  return *this;
}

std::string VectorRef::ToString() const {
  std::ostringstream buffer;
  bool begin = true;
  buffer << "vector[";
  for (auto &attr : elements_) {
    if (!begin) {
      buffer << ", ";
    } else {
      begin = false;
    }
    buffer << attr.ToString();
  }
  buffer << "]";
  return buffer.str();
}

bool VectorRef::operator==(const BaseRef &other) const {
  if (!utils::isa<VectorRef>(other)) {
    return false;
  }
  return *this == utils::cast<VectorRef>(other);
}

bool VectorRef::operator==(const VectorRef &other) const {
  if (elements_.size() != other.elements_.size()) {
    return false;
  }
  for (size_t i = 0; i < elements_.size(); ++i) {
    if (elements_[i] != other.elements_[i]) {
      return false;
    }
  }
  return true;
}

bool SetRef::operator==(const BaseRef &other) const {
  if (!utils::isa<SetRef>(other)) {
    return false;
  }
  return *this == utils::cast<SetRef>(other);
}

bool SetRef::operator==(const SetRef &other) const {
  if (elements_.size() != other.elements_.size()) {
    return false;
  }
  auto iter = elements_.begin();
  auto oth_iter = other.elements_.begin();
  for (; iter != elements_.end(); iter++, oth_iter++) {
    if (*iter != *oth_iter) {
      return false;
    }
  }
  return true;
}

bool RunFunctionRef::operator==(const BaseRef &other) const {
  if (!utils::isa<RunFunctionRef>(other)) {
    return false;
  }
  return *this == utils::cast<RunFunctionRef>(other);
}

bool RunFunctionRef::operator==(const RunFunctionRef &other) const { return func_ == other.func_; }
}  // namespace mindspore
