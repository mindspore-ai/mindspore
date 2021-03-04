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

#ifndef MINDSPORE_CORE_UTILS_SINGLETON_H_
#define MINDSPORE_CORE_UTILS_SINGLETON_H_

namespace mindspore {
template <typename T>
class Singleton {
 public:
  explicit Singleton(T &&) = delete;
  explicit Singleton(const T &) = delete;
  void operator=(const T &) = delete;
  // thread safety implement
  template <typename... _Args>
  static T &Instance(_Args... args) {
    static T instance(args...);
    return instance;
  }

 protected:
  Singleton() = default;
  virtual ~Singleton() = default;
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_UTILS_SINGLETON_H_
