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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_ALLOCATOR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_ALLOCATOR_H_

#include <cstdlib>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include "minddata/dataset/util/memory_pool.h"

namespace mindspore {
namespace dataset {
// The following conforms to the requirements of
// std::allocator. Do not rename/change any needed
// requirements, e.g. function names, typedef etc.
template <typename T>
class Allocator {
 public:
  template <typename U>
  friend class Allocator;

  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;
  using size_type = uint64_t;
  using difference_type = std::ptrdiff_t;

  template <typename U>
  struct rebind {
    using other = Allocator<U>;
  };

  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;

  explicit Allocator(const std::shared_ptr<MemoryPool> &b) : pool_(b) {}

  ~Allocator() = default;

  template <typename U>
  explicit Allocator(Allocator<U> const &rhs) : pool_(rhs.pool_) {}

  template <typename U>
  bool operator==(Allocator<U> const &rhs) const {
    return pool_ == rhs.pool_;
  }

  template <typename U>
  bool operator!=(Allocator<U> const &rhs) const {
    return pool_ != rhs.pool_;
  }

  pointer allocate(std::size_t n) {
    void *p = nullptr;
    Status rc = pool_->Allocate(n * sizeof(T), &p);
    if (rc.IsOk()) {
      return reinterpret_cast<pointer>(p);
    } else if (rc == StatusCode::kMDOutOfMemory) {
      MS_LOG(ERROR) << rc.ToString();
      return nullptr;
    } else {
      MS_LOG(ERROR) << rc.ToString();
      return nullptr;
    }
  }

  void deallocate(pointer p, std::size_t n = 0) noexcept { pool_->Deallocate(p); }

  size_type max_size() { return pool_->get_max_size(); }

 private:
  std::shared_ptr<MemoryPool> pool_;
};
/// \brief It is a wrapper of unique_ptr with a custom Allocator class defined above
template <typename T, typename C = std::allocator<T>, typename... Args>
Status MakeUnique(std::unique_ptr<T[], std::function<void(T *)>> *out, C alloc, size_t n, Args &&... args) {
  RETURN_UNEXPECTED_IF_NULL(out);
  CHECK_FAIL_RETURN_UNEXPECTED(n > 0, "size must be positive");
  T *data = nullptr;
  try {
    data = alloc.allocate(n);
    // Some of our implementation of allocator (e.g. NumaAllocator) don't throw std::bad_alloc.
    // So we have to catch for null ptr
    if (data == nullptr) {
      return Status(StatusCode::kMDOutOfMemory);
    }
    if (!std::is_arithmetic<T>::value) {
      for (size_t i = 0; i < n; i++) {
        std::allocator_traits<C>::construct(alloc, &(data[i]), std::forward<Args>(args)...);
      }
    }
    auto deleter = [](T *p, C f_alloc, size_t f_n) {
      if (!std::is_arithmetic<T>::value && std::is_destructible<T>::value) {
        for (size_t i = 0; i < f_n; ++i) {
          std::allocator_traits<C>::destroy(f_alloc, &p[i]);
        }
      }
      f_alloc.deallocate(p, f_n);
    };
    *out = std::unique_ptr<T[], std::function<void(T *)>>(data, std::bind(deleter, std::placeholders::_1, alloc, n));
  } catch (const std::bad_alloc &e) {
    if (data != nullptr) {
      alloc.deallocate(data, n);
    }
    return Status(StatusCode::kMDOutOfMemory);
  } catch (const std::exception &e) {
    if (data != nullptr) {
      alloc.deallocate(data, n);
    }
    RETURN_STATUS_UNEXPECTED(e.what());
  }
  return Status::OK();
}

/// \brief It is a wrapper of the above custom unique_ptr with some additional methods
/// \tparam T The type of object to be allocated
/// \tparam C Allocator. Default to std::allocator
template <typename T, typename C = std::allocator<T>>
class MemGuard {
 public:
  using allocator = C;
  MemGuard() : n_(0) {}
  explicit MemGuard(const allocator &a) : n_(0), alloc_(a) {}
  // There is no copy constructor nor assignment operator because the memory is solely owned by this object.
  MemGuard(const MemGuard &) = delete;
  MemGuard &operator=(const MemGuard &) = delete;
  // On the other hand, We can support move constructor
  MemGuard(MemGuard &&lhs) noexcept : n_(lhs.n_), alloc_(std::move(lhs.alloc_)), ptr_(std::move(lhs.ptr_)) {}
  MemGuard &operator=(MemGuard &&lhs) noexcept {
    if (this != &lhs) {
      this->deallocate();
      n_ = lhs.n_;
      alloc_ = std::move(lhs.alloc_);
      ptr_ = std::move(lhs.ptr_);
    }
    return *this;
  }
  /// \brief Explicitly deallocate the memory if allocated
  void deallocate() {
    if (ptr_) {
      ptr_.reset();
    }
  }
  /// \brief Allocate memory (with emplace feature). Previous one will be released. If size is 0, no new memory is
  /// allocated.
  /// \param n Number of objects of type T to be allocated
  /// \tparam Args Extra arguments pass to the constructor of T
  template <typename... Args>
  Status allocate(size_t n, Args &&... args) noexcept {
    deallocate();
    n_ = n;
    return MakeUnique(&ptr_, alloc_, n, std::forward<Args>(args)...);
  }
  ~MemGuard() noexcept { deallocate(); }
  /// \brief Getter function
  /// \return The pointer to the memory allocated
  T *GetPointer() const { return ptr_.get(); }
  /// \brief Getter function
  /// \return The pointer to the memory allocated
  T *GetMutablePointer() { return ptr_.get(); }
  /// \brief Overload [] operator to access a particular element
  /// \param x index to the element. Must be less than number of element allocated.
  /// \return pointer to the x-th element
  T *operator[](size_t x) { return GetMutablePointer() + x; }
  /// \brief Overload [] operator to access a particular element
  /// \param x index to the element. Must be less than number of element allocated.
  /// \return pointer to the x-th element
  T *operator[](size_t x) const { return GetPointer() + x; }
  /// \brief Return how many bytes are allocated in total
  /// \return Number of bytes allocated in total
  size_t GetSizeInBytes() const { return n_ * sizeof(T); }

 private:
  size_t n_;
  allocator alloc_;
  std::unique_ptr<T[], std::function<void(T *)>> ptr_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_ALLOCATOR_H_
