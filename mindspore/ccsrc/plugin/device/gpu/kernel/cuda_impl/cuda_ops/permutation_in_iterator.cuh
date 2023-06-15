/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_PERMUTATION_IN_ITERATOR_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_PERMUTATION_IN_ITERATOR_CUH_

#include <iostream>
#include <iterator>


template <typename T, typename IndexIteratorT, typename OffsetT = ptrdiff_t>
class PermutationInputIterator {
 public:
  // Required iterator traits
  typedef PermutationInputIterator self_type;         ///< My own type
  typedef OffsetT difference_type;                    ///< Type to express the result of
                                                      ///< subtracting one iterator from another
  typedef T value_type;                               ///< The type of the element the iterator can point to
  typedef T *pointer;                                 ///< The type of a pointer to an element the
                                                      ///< iterator can point to
  typedef T reference;                                ///< The type of a reference to an element the
                                                      ///< iterator can point to

  typedef std::random_access_iterator_tag iterator_category;  ///< The iterator category

 private:
  const T *in;
  IndexIteratorT index_itr;

 public:
  /// Constructor
  __host__ __device__ __forceinline__
  PermutationInputIterator(const T *in, IndexIteratorT index_itr)
      : in(in), index_itr(index_itr) {}

  /// Postfix increment
  __host__ __device__ __forceinline__ self_type operator++(int) {
    self_type retval = *this;
    index_itr++;
    return retval;
  }

  /// Prefix increment
  __host__ __device__ __forceinline__ self_type operator++() {
    index_itr++;
    return *this;
  }

  /// Indirection
  __host__ __device__ __forceinline__ reference operator*() const { return in[*index_itr]; }

  /// Addition
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator+(Distance n) const {
    self_type retval(in, index_itr + n);
    return retval;
  }

  /// Addition assignment
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type &operator+=(Distance n) {
    index_itr += n;
    return *this;
  }

  /// Subtraction
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator-(Distance n) const {
    self_type retval(in, index_itr - n);
    return retval;
  }

  /// Subtraction assignment
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type &operator-=(Distance n) {
    index_itr -= n;
    return *this;
  }

  /// Distance
  __host__ __device__ __forceinline__ difference_type operator-(self_type other) const {
    return index_itr - other.index_itr;
  }

  /// Array subscript
  template <typename Distance>
  __host__ __device__ __forceinline__ reference operator[](Distance n) const {
    return in[index_itr[n]];
  }

  /// Structure dereference
  __host__ __device__ __forceinline__ pointer operator->() { return in + *index_itr; }

  /// Equal to
  __host__ __device__ __forceinline__ bool operator==(const self_type &rhs) {
    return (index_itr == rhs.index_itr && in == rhs.in);
  }

  /// Not equal to
  __host__ __device__ __forceinline__ bool operator!=(const self_type &rhs) { return !(*this == rhs); }

  /// ostream operator
  friend std::ostream &operator<<(std::ostream &os, const self_type &itr) { return os; }
};


#endif  // TENSORFLOW_CORE_UTIL_PERMUTATION_INPUT_ITERATOR_H_
