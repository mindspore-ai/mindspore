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

#ifndef MINDSPORE_CORE_UTILS_CALLBACK_HANDLER_H_
#define MINDSPORE_CORE_UTILS_CALLBACK_HANDLER_H_

#include <utility>

#define HANDLER_DEFINE(return_type, name, ...)                                                    \
 public:                                                                                          \
  template <typename... Args>                                                                     \
  static return_type name(const Args &... argss) {                                                \
    if (name##_handler_ == nullptr) {                                                             \
      return return_type();                                                                       \
    }                                                                                             \
    return name##_handler_(argss...);                                                             \
  }                                                                                               \
  using name##Handler = std::function<decltype(name<__VA_ARGS__>)>;                               \
  static void Set##name##Handler(name##Handler handler) { name##_handler_ = std::move(handler); } \
                                                                                                  \
 private:                                                                                         \
  inline static name##Handler name##_handler_;
#endif  // MINDSPORE_CORE_UTILS_CALLBACK_HANDLER_H_
