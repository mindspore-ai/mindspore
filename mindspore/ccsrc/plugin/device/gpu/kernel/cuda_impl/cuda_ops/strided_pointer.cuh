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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_STRIDE_POINTER_CUH_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_STRIDE_POINTER_CUH_

#include <limits.h>
#include <cuda_runtime.h>
#include <algorithm>

// ConstStridedPointer is a const random access iterator defined over a strided array.
template <typename T, typename index_t = int64_t>
class ConstStridedPointer {
 public:
  __device__ ConstStridedPointer() : ptr{nullptr}, stride{static_cast<index_t>(1)} {}
  __device__ explicit ConstStridedPointer(T *ptr) : ptr{ptr}, stride{static_cast<index_t>(1)} {}
  __device__ ConstStridedPointer(T *ptr, index_t stride) : ptr{ptr}, stride{stride} {}

  // Pointer-like operations
  __device__ const T &operator[](index_t idx) const { return ptr[idx * stride]; }
  __device__ const T &operator*() const { return *ptr; }
  __device__ const T *operator->() const { return reinterpret_cast<const T *>(ptr); }

  // Prefix/postfix increment/decrement
  __device__ ConstStridedPointer operator++(int) {
    ConstStridedPointer copy(*this);
    ++*this;
    return copy;
  }

  __device__ ConstStridedPointer &operator++() {
    ptr += stride;
    return *this;
  }

  __device__ ConstStridedPointer operator--(int) {
    ConstStridedPointer copy(*this);
    --*this;
    return copy;
  }

  __device__ ConstStridedPointer &operator--() {
    ptr -= stride;
    return *this;
  }

  // Arithmetic operations
  __device__ friend ConstStridedPointer operator+(index_t offset, const ConstStridedPointer &accessor) {
    return accessor + offset;
  }

  __device__ ConstStridedPointer operator+(index_t offset) const {
    return ConstStridedPointer(ptr + offset * stride, stride);
  }

  __device__ ConstStridedPointer &operator+=(index_t offset) {
    ptr += offset * stride;
    return *this;
  }

  __device__ ConstStridedPointer operator-(index_t offset) const {
    return ConstStridedPointer(ptr - offset * stride, stride);
  }

  __device__ ConstStridedPointer &operator-=(index_t offset) {
    ptr -= offset * stride;
    return *this;
  }

  __device__ index_t operator-(const ConstStridedPointer &other) const { return (ptr - other.ptr) / stride; }

  // Comparison operators
  __device__ bool operator>=(const ConstStridedPointer &other) const { return !(*this < other); }
  __device__ bool operator>(const ConstStridedPointer &other) const { return !(*this <= other); }
  __device__ bool operator<=(const ConstStridedPointer &other) const { return (*this < other) || (*this == other); }
  __device__ bool operator<(const ConstStridedPointer &other) const { return ptr < other.ptr; }
  __device__ bool operator!=(const ConstStridedPointer &other) const { return !(*this == other); }
  __device__ bool operator==(const ConstStridedPointer &other) const {
    return (ptr == other.ptr) && (stride == other.stride);
  }

 protected:
  index_t stride;
  T *ptr;
};

// StridedPointer is a random access iterator defined over a strided array.
template <typename T, typename index_t = int64_t>
class StridedPointer : public ConstStridedPointer<T, index_t> {
 public:
  __device__ explicit StridedPointer(T *ptr) : ConstStridedPointer<T, index_t>(ptr) {}
  __device__ StridedPointer(T *ptr, index_t stride) : ConstStridedPointer<T, index_t>(ptr, stride) {}
  __device__ StridedPointer() : ConstStridedPointer<T, index_t>() {}

  // Pointer-like operations
  __device__ T &operator[](index_t idx) const { return this->ptr[idx * this->stride]; }
  __device__ T *operator->() const { return reinterpret_cast<T *>(this->ptr); }
  __device__ T &operator*() const { return *this->ptr; }

  // Prefix/postfix increment/decrement
  __device__ StridedPointer operator++(int) {
    StridedPointer copy(*this);
    ++*this;
    return copy;
  }

  __device__ StridedPointer &operator++() {
    this->ptr += this->stride;
    return *this;
  }

  __device__ StridedPointer operator--(int) {
    StridedPointer copy(*this);
    --*this;
    return copy;
  }

  __device__ StridedPointer &operator--() {
    this->ptr -= this->stride;
    return *this;
  }

  // Arithmetic operations
  __device__ StridedPointer &operator-=(index_t offset) {
    this->ptr -= offset * this->stride;
    return *this;
  }

  __device__ StridedPointer operator-(index_t offset) const {
    return StridedPointer(this->ptr - offset * this->stride, this->stride);
  }

  __device__ friend StridedPointer operator+(index_t offset, const StridedPointer &accessor) {
    return accessor + offset;
  }

  __device__ StridedPointer operator+(index_t offset) const {
    return StridedPointer(this->ptr + offset * this->stride, this->stride);
  }

  __device__ StridedPointer &operator+=(index_t offset) {
    this->ptr += offset * this->stride;
    return *this;
  }

  __device__ index_t operator-(const ConstStridedPointer<T, index_t> &other) const {
    return (static_cast<const ConstStridedPointer<T, index_t> &>(*this) - other);
  }
};

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_STRIDE_POINTER_CUH_
