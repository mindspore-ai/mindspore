/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
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

#ifndef CPU_KERNEL_UTIL_SPARSE_GROUP_ITERATOR_H_
#define CPU_KERNEL_UTIL_SPARSE_GROUP_ITERATOR_H_

#include <vector>
#include "eigen_tensor.h"

namespace aicpu {
class Group;  // Predeclare Group for GroupIterable.

// ///////////////
// GroupIterable
// ///////////////
//
// Returned when calling sparse_tensor.group({dim0, dim1, ...}).
//
// Please note: the sparse_tensor should already be ordered according
// to {dim0, dim1, ...}.  Otherwise this iteration will return invalid groups.
//
// Allows grouping and iteration of the SparseTensor according to the
// subset of dimensions provided to the group call.
//
// The actual grouping dimensions are stored in the
// internal vector group_dims_.  Iterators inside the iterable provide
// the three methods:
//
// *  group(): returns a vector with the current group dimension values.
// *  indices(): a map of index, providing the indices in
//    this group.
// *  values(): a map of values, providing the values in
//    this group.
//
// To iterate across GroupIterable, see examples in README.md.
//

// Forward declaration of SparseTensor
class GroupIterable {
 public:
  using VarDimArray = std::vector<int64_t>;

  GroupIterable(Tensor *ix, Tensor *vals, int dims, const VarDimArray &group_dims)
      : ix_(ix),
        ix_matrix_(EigenTensor(ix, ix->GetData()).matrix<int64_t>()),
        vals_(vals),
        dims_(dims),
        group_dims_(group_dims.begin(), group_dims.end()) {}

  ~GroupIterable() {}

  class IteratorStep;

  IteratorStep begin() { return IteratorStep(this, 0); }

  IteratorStep at(int64_t loc) {
    if (!(loc >= 0 && loc <= static_cast<int64_t>(ix_->GetTensorShape()->GetDimSize(0)))) {
      KERNEL_LOG_WARN("loc should in [0, %d], but got: %d", ix_->GetTensorShape()->GetDimSize(0), loc);
    }
    return IteratorStep(this, loc);
  }

  IteratorStep end() { return IteratorStep(this, ix_->GetTensorShape()->GetDimSize(0)); }

  template <typename TIX>
  inline bool GroupMatches(const TIX &ix, int64_t loc_a, int64_t loc_b) const {
    for (int64_t d : group_dims_) {
      if (ix(loc_a, d) != ix(loc_b, d)) {
        return false;
      }
    }
    return true;
  }

  class IteratorStep {
   public:
    IteratorStep(GroupIterable *iter, int64_t loc) : iter_(iter), loc_(loc), next_loc_(loc_) { UpdateEndOfGroup(); }

    ~IteratorStep() { iter_ = nullptr; }

    void UpdateEndOfGroup();

    bool operator!=(const IteratorStep &rhs) const;

    bool operator==(const IteratorStep &rhs) const;

    IteratorStep &operator++();

    const IteratorStep operator++(int);

    Group operator*() const;

    int64_t loc() const { return loc_; }

   private:
    GroupIterable *iter_;
    int64_t loc_;
    int64_t next_loc_;
  };

 private:
  friend class Group;
  Tensor *ix_;
  TTypes<int64_t>::Matrix ix_matrix_;
  Tensor *vals_;
  const int dims_;
  const std::vector<int64_t> group_dims_;
};

// This class is returned when dereferencing a GroupIterable iterator.
// It provides the methods group(), indices(), and values(), which
// provide access into the underlying SparseTensor.
class Group {
 public:
  Group(GroupIterable *iter, int64_t loc, int64_t next_loc) : iter_(iter), loc_(loc), next_loc_(next_loc) {}

  ~Group() { iter_ = NULL; }

  std::vector<int64_t> group() const;

  TTypes<int64_t>::UnalignedConstMatrix indices() const;

  int64_t group_at(size_t index) const {
    const auto &ix_t = iter_->ix_matrix_;
    return ix_t(loc_, index);
  }

  template <typename T>
  typename TTypes<T>::UnalignedVec values() const {
    return typename TTypes<T>::UnalignedVec(&(EigenTensor(iter_->vals_, iter_->vals_->GetData()).vec<T>()(loc_)),
                                            next_loc_ - loc_);
  }

 private:
  GroupIterable *iter_;
  int64_t loc_;
  int64_t next_loc_;
};
}  // namespace aicpu

#endif  // CPU_KERNEL_UTIL_SPARSE_GROUP_ITERATOR_H_
