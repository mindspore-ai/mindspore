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

#ifndef MINDSPORE_CORE_MINDAPI_BASE_BASE_H_
#define MINDSPORE_CORE_MINDAPI_BASE_BASE_H_

#include <cstdint>
#include <string>
#include <memory>
#include "mindapi/base/macros.h"
#include "mindapi/base/type_traits.h"
#include "mindapi/base/shared_ptr.h"

namespace mindspore {
class Base;
}

namespace mindspore::api {
/// \brief Base is the base class of many api classes, which provides basic interfaces.
class MIND_API Base {
 public:
  /// \brief Create an instance from the given implementation object.
  ///
  /// \param[in] impl The shared_ptr to the implementation object.
  explicit Base(const std::shared_ptr<mindspore::Base> &impl);

  /// \brief Destructor of Base.
  virtual ~Base() = default;

  /// \brief Get the id of this class.
  ///
  /// \return The id of this class.
  static uint32_t ClassId();

  /// \brief Get the shared_ptr to the underly implementation object.
  ///
  /// \return The shared_ptr to the underly implementation object.
  const std::shared_ptr<mindspore::Base> &impl() const { return impl_; }

  /// \brief Get the string representation of this object.
  ///
  /// \return The string representation.
  std::string ToString() const;

  /// \brief Check whether this object is an instance of the given class.
  ///
  /// \return True if this object is an instance of the given class, false otherwise.
  template <typename T, typename = typename std::enable_if_t<std::is_base_of_v<Base, T>, T>>
  inline bool isa() const {
    return IsFromClassId(T::ClassId());
  }

  /// \brief Cast this object to a pointer with the given pointer class.
  ///
  /// \return A non-null pointer if cast success, nullptr otherwise.
  template <typename T, typename U = typename std::enable_if_t<is_wrapper_ptr<T>::value, typename T::element_type>>
  inline T cast() {
    if (isa<U>()) {
      return MakeShared<U>(impl_);
    }
    return nullptr;
  }

 protected:
  bool IsFromClassId(uint32_t class_id) const;
  const std::shared_ptr<mindspore::Base> impl_;
};

#define MIND_API_BASE_MEMBER(current_class)                             \
  explicit current_class(const std::shared_ptr<mindspore::Base> &impl); \
  ~current_class() override = default;                                  \
  static uint32_t ClassId()

using BasePtr = SharedPtr<Base>;
}  // namespace mindspore::api
#endif  // MINDSPORE_CORE_MINDAPI_BASE_BASE_H_
