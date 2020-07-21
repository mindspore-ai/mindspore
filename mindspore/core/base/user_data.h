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

#ifndef MINDSPORE_CORE_USER_DATA_H_
#define MINDSPORE_CORE_USER_DATA_H_

#include <string>
#include <memory>
#include <map>

namespace mindspore {
class UserData {
 public:
  template <typename T>
  void set(const std::string &key, const std::shared_ptr<T> &value) {
    if (value == nullptr) {
      data_.erase(key);
    } else {
      data_.insert_or_assign(key, value);
    }
  }

  template <typename T>
  std::shared_ptr<T> get(const std::string &key) const {
    auto iter = data_.find(key);
    if (iter == data_.end()) {
      return nullptr;
    }
    return std::static_pointer_cast<T>(iter->second);
  }

  bool has(const std::string &key) const { return data_.find(key) != data_.end(); }

 private:
  std::map<std::string, std::shared_ptr<void>> data_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_USER_DATA_H_
