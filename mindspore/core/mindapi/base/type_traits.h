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

#ifndef MINDSPORE_CORE_MINDAPI_BASE_TYPE_TRAITS_H_
#define MINDSPORE_CORE_MINDAPI_BASE_TYPE_TRAITS_H_

#include <vector>
#include <memory>
#include <type_traits>
#include "mindapi/base/shared_ptr.h"

namespace mindspore::api {
template <typename T>
struct is_wrapper_ptr : public std::false_type {};
template <typename T>
struct is_wrapper_ptr<SharedPtr<T>> : public std::true_type {};

template <typename T>
struct is_shared_ptr : public std::false_type {};
template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : public std::true_type {};

template <typename T>
struct is_vector : public std::false_type {};
template <typename T, typename A>
struct is_vector<std::vector<T, A>> : public std::true_type {};
}  // namespace mindspore::api

#endif  // MINDSPORE_CORE_MINDAPI_BASE_TYPE_TRAITS_H_
