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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_CYCLIC_ARRAY_H
#define MINDSPORE_CCSRC_MINDDATA_DATASET_CYCLIC_ARRAY_H

#include <memory>
#include <algorithm>
#include <cstring>
#include <type_traits>
#include "minddata/dataset/include/constants.h"

namespace mindspore {
namespace dataset {

/// \class CyclicArray "include/cyclic_array.h
/// \brief This is a container with a contiguous memory layout that pnly keeps N last entries,
///        when the number of entries exceeds the capacity
///        Must be preallocated
template <typename T>
class CyclicArray {
 public:
  using value_type = T;
  class Iterator {
    // Add operator[] and make fully compliant with random access iterator
    // and add a const iterator
    // add resize(), empty()
   public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = CyclicArray::value_type;
    using difference_type = std::ptrdiff_t;
    using pointer = CyclicArray::value_type *;
    using reference = CyclicArray::value_type &;

    Iterator() = default;

    Iterator(dsize_t idx, pointer ptr, dsize_t capacity, dsize_t head)
        : cur_idx_(idx), ptr_(ptr), capacity_(capacity), head_(head) {}

    Iterator(const Iterator &rhs) = default;

    ~Iterator() = default;

    Iterator &operator++() {
      cur_idx_ = (cur_idx_ + 1) % (capacity_ + 1);
      return *this;
    }

    Iterator operator++(int) {
      Iterator tmp(*this);
      cur_idx_ = (cur_idx_ + 1) % (capacity_ + 1);
      return tmp;
    }

    Iterator &operator--() {
      cur_idx_ = (cur_idx_ + capacity_) % (capacity_ + 1);
      return *this;
    }

    Iterator operator--(int) {
      Iterator tmp(*this);
      cur_idx_ = (cur_idx_ + capacity_) % (capacity_ + 1);
      return tmp;
    }

    Iterator operator+(dsize_t x) { return Iterator((cur_idx_ + x) % (capacity_ + 1), ptr_, capacity_, head_); }

    Iterator operator-(dsize_t x) {
      return Iterator((cur_idx_ + (capacity_ + 1 - x)) % (capacity_ + 1), ptr_, capacity_, head_);
    }

    bool operator<(const Iterator &rhs) {
      return (head_ + cur_idx_) % (capacity_ + 1) < (rhs.head_ + rhs.cur_idx_) % (capacity_ + 1);
    }

    bool operator>(const Iterator &rhs) {
      return (head_ + cur_idx_) % (capacity_ + 1) > (rhs.head_ + rhs.cur_idx_) % (capacity_ + 1);
    }

    bool operator>=(const Iterator &rhs) {
      return (head_ + cur_idx_) % (capacity_ + 1) >= (rhs.head_ + rhs.cur_idx_) % (capacity_ + 1);
    }

    bool operator<=(const Iterator &rhs) {
      return (head_ + cur_idx_) % (capacity_ + 1) <= (rhs.head_ + rhs.cur_idx_) % (capacity_ + 1);
    }

    difference_type operator-(const Iterator &rhs) {
      return (cur_idx_ - rhs.cur_idx_ + capacity_ + 1) % (capacity_ + 1);
    }

    reference operator*() { return ptr_[cur_idx_]; }

    pointer operator->() { return &(ptr_[cur_idx_]); }

    bool operator==(const Iterator &rhs) { return cur_idx_ == rhs.cur_idx_; }

    bool operator!=(const Iterator &rhs) { return cur_idx_ != rhs.cur_idx_; }

   private:
    dsize_t cur_idx_;
    pointer ptr_;
    dsize_t capacity_;
    dsize_t head_;
  };

  /// \brief Default constructor
  CyclicArray() : buf_(nullptr), head_(0), tail_(0), size_(0), capacity_(0) {}

  /// \brief Constructor
  /// \param[in] capacity
  explicit CyclicArray(dsize_t capacity)
      : buf_(std::make_unique<T[]>(capacity + 1)), head_(0), tail_(0), size_(0), capacity_(capacity) {}

  CyclicArray(const CyclicArray<T> &rhs)
      : buf_(std::make_unique<T[]>(rhs.capacity_ + 1)),
        head_(rhs.head_),
        tail_(rhs.tail_),
        size_(rhs.size_),
        capacity_(rhs.capacity_) {
    std::copy(rhs.begin(), rhs.end(), begin());
  }

  CyclicArray(CyclicArray &&rhs) = default;

  ~CyclicArray() = default;

  /// \brief Iterator begin()
  Iterator begin() { return Iterator(head_, buf_.get(), capacity_, head_); }

  /// \brief Iterator end()
  Iterator end() { return Iterator(tail_, buf_.get(), capacity_, head_); }

  // not really const.
  Iterator begin() const { return Iterator(head_, buf_.get(), capacity_, head_); }

  Iterator end() const { return Iterator(tail_, buf_.get(), capacity_, head_); }

  /// \brief clear the array. Does not deallocate memory, capacity remains the same
  void clear() {
    head_ = 0;
    tail_ = 0;
    size_ = 0;
  }

  /// \brief returns current size
  dsize_t size() { return size_; }

  /// \brief returns capacity
  dsize_t capacity() { return capacity_; }

  /// \brief pushes a value
  /// \param[in] val value
  void push_back(T val) {
    buf_[tail_] = val;
    if (size_ >= capacity_) {
      (tail_ != capacity_) ? tail_++ : tail_ = 0;
      (head_ != capacity_) ? head_++ : head_ = 0;
    } else {
      tail_++;
      size_++;
    }
  }

  /// \brief returns const reference to an element of the array
  /// \param[in] idx index of the element
  /// \param[out] const T& reference to an element of the array
  const T &operator[](dsize_t idx) const { return buf_[(head_ + idx) % (capacity_ + 1)]; }

  /// \brief returns non-const reference to an element of the array
  /// \param[in] idx index of the element
  /// \param[out] T& reference to an element of the array
  T &operator[](dsize_t idx) { return buf_[(head_ + idx) % (capacity_ + 1)]; }

 private:
  std::unique_ptr<T[]> buf_;
  dsize_t head_;
  dsize_t tail_;
  dsize_t size_;
  dsize_t capacity_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CYCLIC_ARRAY_H
