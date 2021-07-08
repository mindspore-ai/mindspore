/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FL_SERVER_LOCAL_META_STORE_H_
#define MINDSPORE_CCSRC_FL_SERVER_LOCAL_META_STORE_H_

#include <any>
#include <mutex>
#include <string>
#include <unordered_map>
#include "fl/server/common.h"

namespace mindspore {
namespace fl {
namespace server {
// LocalMetaStore class is used for metadata storage of this server process.
// For example, the current iteration number, time windows for round kernels, etc.
// LocalMetaStore is threadsafe.
class LocalMetaStore {
 public:
  static LocalMetaStore &GetInstance() {
    static LocalMetaStore instance;
    return instance;
  }

  template <typename T>
  void put_value(const std::string &name, const T &value) {
    std::unique_lock<std::mutex> lock(mtx_);
    key_to_meta_[name] = value;
  }

  template <typename T>
  T value(const std::string &name) {
    std::unique_lock<std::mutex> lock(mtx_);
    try {
      T value = std::any_cast<T>(key_to_meta_[name]);
      return value;
    } catch (const std::exception &e) {
      MS_LOG(EXCEPTION) << "Value of " << name << " is not set.";
    }
  }

  // This method returns a reference so that user can change this value without calling put_value.
  template <typename T>
  T &mutable_value(const std::string &name) {
    std::unique_lock<std::mutex> lock(mtx_);
    try {
      return std::any_cast<T &>(key_to_meta_[name]);
    } catch (const std::exception &e) {
      MS_LOG(EXCEPTION) << "Value of " << name << " is not set.";
    }
  }

  void remove_value(const std::string &name);
  bool has_value(const std::string &name);

  void set_curr_iter_num(size_t num);
  const size_t curr_iter_num();

 private:
  LocalMetaStore() : key_to_meta_({}), curr_iter_num_(0) {}
  ~LocalMetaStore() = default;
  LocalMetaStore(const LocalMetaStore &) = delete;
  LocalMetaStore &operator=(const LocalMetaStore &) = delete;

  // key_to_meta_ stores metadata with key-value format.
  std::unordered_map<std::string, std::any> key_to_meta_;
  // This mutex makes sure that the operations on key_to_meta_ is threadsafe.
  std::mutex mtx_;
  size_t curr_iter_num_;
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_LOCAL_META_STORE_H_
