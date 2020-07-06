/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef DATASET_UTIL_SLICE_H_
#define DATASET_UTIL_SLICE_H_

#include <unistd.h>
#include <cstddef>
#include <utility>
#include "./securec.h"
#include "dataset/util/allocator.h"
#include "dataset/util/status.h"
namespace mindspore {
namespace dataset {
/// \brief A ReadableSlice wraps a const pointer in memory and its size.
/// \see WritableSlice for a non-const version
///
class ReadableSlice {
 public:
  ReadableSlice() : ptr_(nullptr), sz_(0) {}
  ReadableSlice(const void *ptr, size_t sz) : ptr_(ptr), sz_(sz) {}
  ReadableSlice(const ReadableSlice &src, off64_t offset, size_t len) {
    ptr_ = static_cast<const char *>(src.GetPointer()) + offset;
    sz_ = len;
  }
  ReadableSlice(const ReadableSlice &src, off64_t offset) : ReadableSlice(src, offset, src.sz_ - offset) {}
  ReadableSlice(const ReadableSlice &lhs) {
    ptr_ = lhs.ptr_;
    sz_ = lhs.sz_;
  }
  ReadableSlice &operator=(const ReadableSlice &lhs) {
    if (this != &lhs) {
      ptr_ = lhs.ptr_;
      sz_ = lhs.sz_;
    }
    return *this;
  }
  ReadableSlice(ReadableSlice &&lhs) noexcept {
    if (this != &lhs) {
      ptr_ = lhs.ptr_;
      sz_ = lhs.sz_;
      lhs.ptr_ = nullptr;
      lhs.sz_ = 0;
    }
  }
  ReadableSlice &operator=(ReadableSlice &&lhs) noexcept {
    if (this != &lhs) {
      ptr_ = lhs.ptr_;
      sz_ = lhs.sz_;
      lhs.ptr_ = nullptr;
      lhs.sz_ = 0;
    }
    return *this;
  }
  /// \brief Getter function
  /// \return Const version of the pointer
  const void *GetPointer() const { return ptr_; }
  /// \brief Getter function
  /// \return Size of the slice
  size_t GetSize() const { return sz_; }
  bool empty() const { return ptr_ == nullptr; }

 private:
  const void *ptr_;
  size_t sz_;
};
/// \brief A WritableSlice inherits from ReadableSlice to allow
/// one to write to the address pointed to by the pointer.
///
class WritableSlice : public ReadableSlice {
 public:
  friend class StorageContainer;
  /// \brief Default constructor
  WritableSlice() : ReadableSlice(), mutable_data_(nullptr) {}
  /// \brief This form of a constructor takes a pointer and its size.
  WritableSlice(void *ptr, size_t sz) : ReadableSlice(ptr, sz), mutable_data_(ptr) {}
  WritableSlice(const WritableSlice &src, off64_t offset, size_t len);
  WritableSlice(const WritableSlice &src, off64_t offset);
  WritableSlice(const WritableSlice &lhs) : ReadableSlice(lhs) { mutable_data_ = lhs.mutable_data_; }
  WritableSlice &operator=(const WritableSlice &lhs) {
    if (this != &lhs) {
      mutable_data_ = lhs.mutable_data_;
      ReadableSlice::operator=(lhs);
    }
    return *this;
  }
  WritableSlice(WritableSlice &&lhs) noexcept : ReadableSlice(std::move(lhs)) {
    if (this != &lhs) {
      mutable_data_ = lhs.mutable_data_;
      lhs.mutable_data_ = nullptr;
    }
  }
  WritableSlice &operator=(WritableSlice &&lhs) noexcept {
    if (this != &lhs) {
      mutable_data_ = lhs.mutable_data_;
      lhs.mutable_data_ = nullptr;
      ReadableSlice::operator=(std::move(lhs));
    }
    return *this;
  }
  /// \brief Copy the content from one slice onto another.
  static Status Copy(WritableSlice *dest, const ReadableSlice &src);

 private:
  void *mutable_data_;
  void *GetMutablePointer() { return mutable_data_; }
};
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_UTIL_SLICE_H_
