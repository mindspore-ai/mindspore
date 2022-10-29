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

#ifndef MINDSPORE_CORE_USER_DATA_H_
#define MINDSPORE_CORE_USER_DATA_H_

#include <string>
#include <memory>
#include <utility>
#include "utils/hash_map.h"

namespace mindspore {
class UserData {
 public:
  using DataMap = mindspore::HashMap<std::string, std::shared_ptr<void>>;

  UserData() = default;
  UserData(const UserData &other) : data_(other.data_ ? std::make_unique<DataMap>(*other.data_) : nullptr) {}
  UserData(UserData &&other) : data_(std::move(other.data_)) {}
  UserData &operator=(const UserData &other) {
    if (this == &other) {
      return *this;
    }
    data_ = (other.data_ ? std::make_unique<DataMap>(*other.data_) : nullptr);
    return *this;
  }
  UserData &operator=(UserData &&other) {
    if (this == &other) {
      return *this;
    }
    data_ = std::move(other.data_);
    return *this;
  }
  ~UserData() = default;

  template <typename T>
  void set(const std::string &key, const std::shared_ptr<T> &value) {
    InitData();
    if (value == nullptr) {
      (void)data_->erase(key);
    } else {
      (void)data_->insert_or_assign(key, value);
    }
  }

  template <typename T>
  std::shared_ptr<T> get(const std::string &key) const {
    if (data_ == nullptr) {
      return nullptr;
    }
    auto iter = data_->find(key);
    if (iter == data_->end()) {
      return nullptr;
    }
    return std::static_pointer_cast<T>(iter->second);
  }

  bool has(const std::string &key) const { return (data_ != nullptr) && (data_->find(key) != data_->end()); }

 private:
  void InitData() {
    if (data_ == nullptr) {
      data_ = std::make_unique<DataMap>();
    }
  }
  std::unique_ptr<DataMap> data_;
};

// User data key name.
constexpr auto kUserDataData = "user_data_data";
constexpr auto kUserDataType = "user_data_type";
constexpr auto kHashTableKeyType = "hash_table_key_type";
constexpr auto kHashTableValueType = "hash_table_value_type";
constexpr auto kHashTableShapeVector = "hash_table_shape_vector";
constexpr auto kHashTableDefaultValue = "hash_table_default_value";

enum class UserDataType { kUserDataTypeUnknown = 0, kUserTypeHashTable };
}  // namespace mindspore

#endif  // MINDSPORE_CORE_USER_DATA_H_
