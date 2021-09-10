/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_INCLUDE_API_ALLOCATOR_H
#define MINDSPORE_INCLUDE_API_ALLOCATOR_H

#include <memory>
#include "include/api/types.h"

namespace mindspore {
/// \brief Allocator defined a memory pool for malloc memory and free memory dynamically.
class MS_API Allocator {
 public:
  /// \brief Destructor of MindSpore Allocator.
  virtual ~Allocator() = default;

  /// \brief Method to request memory.
  ///
  /// \param[in] size Define the memory size to request.
  virtual void *Malloc(size_t size) = 0;

  /// \brief Method to request memory.
  ///
  /// \param[in] weight Defines the width of memory to request
  /// \param[in] height Defines the height of memory to request
  /// \param[in] type Defines the data type of memory to request
  virtual void *Malloc(size_t weight, size_t height, DataType type) {
    return nullptr;
  }

  /// \brief Method to free memory.
  ///
  /// \param[in] ptr Define the pointer of a certain memory.
  virtual void Free(void *ptr) = 0;

  /// \brief Reference count of a certain memory.
  ///
  /// \param[in] ptr Define the pointer of a certain memory.
  ///
  /// \return Reference count of a certain memory currently.
  virtual int RefCount(void *ptr) = 0;

  /// \brief Set reference count of a certain memory.
  ///
  /// \param[in] ptr Define the pointer of a certain memory.
  /// \param[in] ref_count Define the reference count to set.
  ///
  /// \return Reference count of a certain memory after setting.
  virtual int SetRefCount(void *ptr, int ref_count) = 0;

  /// \brief Decrease the reference count of a certain memory.
  ///
  /// \param[in] ptr Define the pointer of a certain memory.
  /// \param[in] ref_count Define the reference count to reduce.
  ///
  /// \return Reference count of a certain memory after decreating.
  virtual int DecRefCount(void *ptr, int ref_count) = 0;

  /// \brief Increase the reference count of a certain memory.
  ///
  /// \param[in] ptr Define the pointer of a certain memory.
  /// \param[in] ref_count Define the reference count to increase.
  ///
  /// \return Reference count of a certain memory after increasing.
  virtual int IncRefCount(void *ptr, int ref_count) = 0;

  /// \brief Static method to create an allocator.
  ///
  /// \return Smart pointer of an allocator.
  static std::shared_ptr<Allocator> Create();

  /// \brief Prepare a certain memory.
  ///
  /// \param[in] ptr Define the pointer of a certain memory to prepare.
  ///
  /// \return Pointer of ready memory.
  virtual void *Prepare(void *ptr) { return ptr; }

 protected:
  // memory aligned bytes
  size_t aligned_size_ = 32;
};
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_ALLOCATOR_H
