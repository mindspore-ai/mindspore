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

#ifndef MINDSPORE_CORE_MINDAPI_IR_UTILS_H_
#define MINDSPORE_CORE_MINDAPI_IR_UTILS_H_

#include "mindapi/base/base.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/base/type_traits.h"
#include "mindapi/ir/anf.h"
#include "mindapi/ir/value.h"
#include "mindapi/ir/func_graph.h"

namespace mindspore::api::utils {
/// \brief Check whether the given object is an instance of the given class.
///
/// \param[in] ptr The pointer to the given object.
///
/// \return True if the pointer is not null and the object is an instance of the given class, false otherwise.
template <typename T, typename = typename std::enable_if_t<std::is_base_of_v<Base, T> || is_wrapper_ptr<T>::value>>
inline bool isa(const BasePtr &ptr) {
  if (ptr == nullptr) {
    return false;
  }
  if constexpr (is_wrapper_ptr<T>::value) {
    return ptr->isa<typename T::element_type>();
  } else {
    return ptr->isa<T>();
  }
}

/// \brief Check whether the given object is an value of the given c++ type T.
///
/// \param[in] ptr The pointer to the given value object.
///
/// \return True if the pointer is not null and it is an value of the given c++ type, false otherwise.
template <typename T, typename U = typename ImmTrait<T>::type::element_type>
inline bool isa(const ValuePtr &ptr) {
  if (ptr == nullptr) {
    return false;
  }
  return ptr->isa<U>();
}

/// \brief Cast the given object pointer to a pointer with the given class.
///
/// \param[in] ptr The pointer to the object to casted.
///
/// \return A non-null pointer if the input pointer is not null and cast success, nullptr otherwise.
template <typename T, typename = typename std::enable_if_t<is_wrapper_ptr<T>::value, T>>
inline T cast(const BasePtr &ptr) {
  if (ptr == nullptr) {
    return nullptr;
  }
  return ptr->cast<T>();
}

/// \brief Cast the given value to a C++ value.
///
/// \param[in] ptr The pointer to the value to be casted.
///
/// \return The C++ value according the input value.
template <typename T, typename U = typename ImmTrait<T>::type>
inline T cast(const ValuePtr &ptr) {
  return GetValue<T>(ptr);
}

/// \brief Make a copy from the given function graph.
///
/// \param[in] func_graph The graph to be cloned.
///
/// \return The cloned graph.
MIND_API FuncGraphPtr CloneGraph(const FuncGraphPtr &func_graph);

/// \brief Get pad mode id from a value holds the pad mode name or id.
///
/// \param[in] value The value holds the pad mode name or id.
/// \param[in] is_upper Indicates whether the name is uppercase or lowercase, default is false for lowercase.
///
/// \return The pad mode id.
MIND_API int64_t GetPadMode(const ValuePtr &value, bool is_upper = false);
}  // namespace mindspore::api::utils

#endif  // MINDSPORE_CORE_MINDAPI_IR_UTILS_H_
