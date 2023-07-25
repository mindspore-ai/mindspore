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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_TRANSFORM_OUTPUT_ITERATOR_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_TRANSFORM_OUTPUT_ITERATOR_CUH_

#include <iostream>
#include <iterator>

template <typename OutputType,    ///< The type of element after mapping
          typename ConversionOp,  ///< Unary functor type for mapping objects of type OriginType to type OutputType
                                  ///< Must have member " OutputType operator()(const OriginType &x)"
          typename OriginType,    ///< The original type of wrapped input iterator
          typename OffsetT = ptrdiff_t>  ///< the difference type of this iterator (Default: ptrdiff_t)
class TransformOutputIterator {
 protected:
  /// Proxy object
  struct Reference {
    OutputType *output_ptr;
    ConversionOp conversion_op;

    /// Constructor
    __host__ __device__ __forceinline__ Reference(OutputType *output_ptr,      ///< Output iterator after wrapping
                                                  ConversionOp conversion_op)  ///< Conversion functor to wrap
        : output_ptr(output_ptr), conversion_op(conversion_op) {}

    /// Assignment
    __host__ __device__ __forceinline__ OriginType operator=(OriginType val) {
      *output_ptr = conversion_op(val);
      return val;
    }
  };

 public:
  /// Required iterator traits
  typedef TransformOutputIterator self_type;  ///< My own type
  typedef OffsetT difference_type;            ///< Type to express the result of subtracting one iterator from another

  typedef void value_type;  ///< In "cub/agent/agent_reduce.cuh"
                            ///< OutputT = (if output iterator's value type if void) ?
                            ///< then the input iterator's value type,
                            ///< else the output iterator's value type
  typedef void pointer;
  typedef Reference reference;  ///< The type of a reference to an element the iterator can point to
  typedef std::random_access_iterator_tag iterator_category;  ///< The iterator category

  OutputType *output_ptr;
  ConversionOp conversion_op;

 public:
  /// Constructor
  template <typename T>
  __host__ __device__ __forceinline__ TransformOutputIterator(T *output_ptr, ConversionOp conversion_op)
      : output_ptr(output_ptr), conversion_op(conversion_op) {}

  /// Postfix increment
  __host__ __device__ __forceinline__ self_type operator++(int) {
    self_type retval = *this;
    output_ptr++;
    return retval;
  }

  /// Prefix increment
  __host__ __device__ __forceinline__ self_type operator++() {
    output_ptr++;
    return *this;
  }

  /// Indirection
  __host__ __device__ __forceinline__ reference operator*() const { return Reference(output_ptr, conversion_op); }

  /// Addition
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator+(Distance n) const {
    self_type retval(output_ptr + n, conversion_op);
    return retval;
  }

  /// Addition assignment
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type &operator+=(Distance n) {
    output_ptr += n;
    return *this;
  }

  /// Subtraction
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator-(Distance n) const {
    self_type retval(output_ptr - n, conversion_op);
    return retval;
  }

  /// Subtraction assignment
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type &operator-=(Distance n) {
    output_ptr -= n;
    return *this;
  }

  /// Distance
  __host__ __device__ __forceinline__ difference_type operator-(self_type other) const {
    return output_ptr - other.output_ptr;
  }

  /// Array subscript
  template <typename Distance>
  __host__ __device__ __forceinline__ reference operator[](Distance n) const {
    return Reference(output_ptr + n, conversion_op);
  }

  /// Equal to
  __host__ __device__ __forceinline__ bool operator==(const self_type &rhs) { return (output_ptr == rhs.output_ptr); }

  /// Not equal to
  __host__ __device__ __forceinline__ bool operator!=(const self_type &rhs) { return (output_ptr != rhs.output_ptr); }

  /// ostream operator
  friend std::ostream &operator<<(std::ostream &os, const self_type &itr) { return os; }
};

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_TRANSFORM_OUTPUT_ITERATOR_CUH_
