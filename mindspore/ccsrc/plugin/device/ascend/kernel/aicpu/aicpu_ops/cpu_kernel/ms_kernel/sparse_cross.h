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
#ifndef AICPU_KERNELS_NORMALIZED_SPARSECROSS_H_
#define AICPU_KERNELS_NORMALIZED_SPARSECROSS_H_

#include <algorithm>
#include <numeric>
#include <vector>
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace swap {
#define STATIC_INLINE static inline
#define BSWAP_8(x) ((x)&0xff)
#define BSWAP_16(x) ((BSWAP_8(x) << 8) | BSWAP_8((x) >> 8))
#define BSWAP_32(x) ((BSWAP_16(x) << 16) | BSWAP_16((x) >> 16))
#define BSWAP_64(x) ((BSWAP_32(x) << 32) | BSWAP_32((x) >> 32))
#define uint32_in_expected_order(x) (x)
#define uint64_in_expected_order(x) (BSWAP_64(x))
}  // namespace swap

namespace aicpu {
class SparseCrossCpuKernel : public CpuKernel {
 public:
  SparseCrossCpuKernel() = default;
  ~SparseCrossCpuKernel() override = default;

 protected:
  // template <bool HASHED_OUTPUT, typename InternalType>
  uint32_t Compute(CpuKernelContext &ctx);

 private:
  template <bool HASHED_OUTPUT, typename InternalType>
  uint32_t SparseCrossCompute(CpuKernelContext &ctx);

  int64_t num_buckets_;
  uint64_t hash_key_;
};

template <typename ListType, typename ElementType>
class OpArgIterator {
 public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = ElementType;
  using pointer = ElementType *;
  using const_pointer = const ElementType *;
  using reference = ElementType &;
  using const_reference = const ElementType &;
  using difference_type = ptrdiff_t;

  OpArgIterator(const ListType *list, int i) : list_(list), i_(i) {}

  bool operator==(const OpArgIterator &rhs) {
    if (list_ == rhs.list_) {
      return i_ == rhs.i_;
    }
    return false;
  }

  bool operator!=(const OpArgIterator &rhs) {
    if (list_ == rhs.list_) {
      return i_ != rhs.i_;
    }
    return true;
  }

  OpArgIterator operator++() {  // prefix ++it
    ++i_;
    return *this;
  }

  OpArgIterator operator++(int) {  // postfix it++
    OpArgIterator old_value = *this;
    ++i_;
    return old_value;
  }

  reference operator*() { return (*list_)[i_]; }
  pointer operator->() { return &(*list_)[i_]; }

  const_reference operator*() const { return (*list_)[i_]; }
  const_pointer operator->() const { return &(*list_)[i_]; }

 private:
  const ListType *const list_;
  int i_;
};

class OpInputList {
 public:
  using Iterator = OpArgIterator<OpInputList, const Tensor>;
  OpInputList() : ctx_(nullptr), start_(0), stop_(0) {}
  OpInputList(CpuKernelContext *ctx, uint32_t start, uint32_t stop) : ctx_(ctx), start_(start), stop_(stop) {}
  OpInputList &operator=(const OpInputList &other) = default;
  OpInputList(const OpInputList &other) = default;
  Tensor *operator[](uint32_t i) const { return ctx_->Input(start_ + i); }
  uint32_t size() const { return stop_ - start_; }
  Iterator begin() const { return Iterator(this, 0); }
  Iterator end() const { return Iterator(this, size()); }

 private:
  CpuKernelContext *ctx_;  // not owned
  uint32_t start_;
  uint32_t stop_;
};
}  // namespace aicpu
#endif