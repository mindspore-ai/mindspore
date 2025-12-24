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

#ifndef MINDSPORE_CORE_OPS_NORMALIZE_SLICE_H_
#define MINDSPORE_CORE_OPS_NORMALIZE_SLICE_H_

#include <limits>
#include <vector>
#include <algorithm>

#include "ops/base_operator.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
constexpr auto kNameNormalizeSlice = "NormalizeSlice";
/// \brief Normalize Slice index info start, stop, step when data shape is dynamic.
// input: data_shape, init_by_none, start, stop, step
// attr: tuple_index_types, dim_axis, expand_dims_mask(used in setitem)
// outputs: start, stop, step
class OPS_API NormalizeSlice : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(NormalizeSlice);
  /// \brief Constructor.
  NormalizeSlice() : BaseOperator(kNameNormalizeSlice) {}
  /// \brief Init function.
  void Init() const {}
};

const int64_t kIndexMax = std::numeric_limits<int64_t>::max();
class IndexSlice {
 public:
  IndexSlice(const int64_t &start_index, const int64_t &stop_index, const int64_t &step_index, int64_t dim_size,
             const std::vector<int64_t> int_by_none, bool slice_to_indices) {
    dim_size_ = dim_size;
    bool step_by_none_init = int_by_none[kIndex2];
    if (step_by_none_init) {
      step_ = 1;
    } else {
      step_ = step_index;
    }
    if (step_ == 0) {
      MS_EXCEPTION(ValueError) << "For 'StridedSlice', 'strides' cannot contain 0";
    }
    if (step_ < -kIndexMax) {
      step_ = -kIndexMax;
    }
    start_ = NormalizeIndex(start_index, step_, dim_size_, int_by_none[0]);
    stop_ = NormalizeIndex(stop_index, -step_, dim_size_, int_by_none[1]);
    is_empty_slice_ = (start_ - stop_) * step_ >= 0;
    if (!slice_to_indices) {
      bool start_init_by_none = int_by_none[0];
      bool stop_init_by_none = int_by_none[1];
      int64_t start_default;
      int64_t stop_default;
      if (step_ < 0) {
        start_default = -1;
        stop_default = -(dim_size_ + 1);
        stop_ = stop_init_by_none ? stop_default : std::max(stop_default, stop_index);
      } else {
        start_default = 0;
        stop_default = dim_size_;
        stop_ = stop_init_by_none ? stop_default : std::min(stop_default, stop_index);
      }
      start_ = start_init_by_none ? start_default : start_index;
    }
  }

  int64_t start() const { return start_; }
  int64_t stop() const { return stop_; }
  int64_t step() const { return step_; }
  bool is_empty_slice() const { return is_empty_slice_; }

  static inline int64_t NormalizeIndex(int64_t index, int64_t dim_size) {
    int64_t new_index = index;
    if (new_index < 0) {
      MS_EXCEPTION_IF_ZERO("DimsSize should not be zero", dim_size);
      return new_index < -dim_size ? 0 : (dim_size + (new_index % dim_size)) % dim_size;  // NOLINT
    }
    return new_index < dim_size ? new_index : dim_size;
  }

  static inline int64_t NormalizeIndex(int64_t index, int64_t step, int64_t dim_size, bool is_none) {
    int64_t normalized_index;
    if (is_none) {
      normalized_index = step > 0 ? 0 : dim_size;
    } else {
      normalized_index = NormalizeIndex(index, dim_size);
    }
    return normalized_index;
  }

 private:
  int64_t start_ = 0;
  int64_t stop_ = 0;
  int64_t step_ = 0;
  int64_t dim_size_ = 0;
  bool is_empty_slice_ = false;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_NORMALIZE_SLICE_H_
