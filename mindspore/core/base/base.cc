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

#include "base/base.h"

namespace mindspore {
Base::Base(const Base &other) : std::enable_shared_from_this<Base>(other) {}

bool Base::operator==(const Base &rhs) {
  if (this == &rhs) {
    return true;
  }
  return false;
}

Base &Base::operator=(const Base &other) {
  if (this == &other) {
    return *this;
  }
  user_data_ = other.user_data_;
  return *this;
}

std::size_t Base::hash() const { return tid(); }

std::string Base::ToString() const { return type_name(); }

void Base::dump() const { std::cout << ToString() << std::endl; }

std::string Base::DumpText() const { return ToString(); }

bool Base::IsFromTypeId(uint32_t tid) const { return Base::IsDerivedFrom(tid); }

bool Base::IsSameTypeId(uint32_t tid) const { return tid == Base::kTypeId; }

std::string Base::type_name() const { return "Base"; }

uint32_t Base::tid() const { return Base::kTypeId; }

bool Base::IsDerivedFrom(uint32_t tid) { return tid == Base::kTypeId; }

bool Base::has_user_data(const std::string &key) const { return user_data_.has(key); }

void Base::CloneUserData(const std::shared_ptr<Base> &other) { user_data_ = other->user_data_; }

void Base::CloneUserData(const UserData &other) { user_data_ = other; }

const UserData &Base::GetUserData() const { return user_data_; }
}  // namespace mindspore
