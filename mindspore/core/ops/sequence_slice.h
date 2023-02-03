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

#ifndef MINDSPORE_CORE_OPS_SEQUENCE_SLICE_H_
#define MINDSPORE_CORE_OPS_SEQUENCE_SLICE_H_

#include "ops/base_operator.h"
#include "mindspore/core/ops/core_ops.h"

namespace mindspore {
namespace ops {
/// \brief Sequence slice operation.
class MIND_API SequenceSlice : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SequenceSlice);
  /// \brief Constructor.
  SequenceSlice() : BaseOperator(prim::kSequenceSlice) {}
  /// \brief Init function.
  void Init() const {}
};

inline static int64_t SequenceSliceGetOutputSize(int64_t start, int64_t stop, int64_t step, int64_t len) {
  int64_t idx = 0;
  if (step > 0) {
    if (start <= -len) {
      start = 0;
    } else if (start < 0) {
      start += len;
    }
    if (stop > len) {
      stop = len;
    } else if (stop > -len && stop < 0) {
      stop += len;
    }
    if (start >= stop) {
      return 0;
    }
    for (int i = start; i < stop; i += step) {
      idx++;
    }
  }

  if (step < 0) {
    if (start >= len) {
      start = -1;
    } else if (start >= 0 && start < len) {
      start -= len;
    }
    if (stop < -len) {
      stop = -1 - len;
    } else if (stop >= 0 && stop < len) {
      stop -= len;
    }
    if (start <= stop) {
      return 0;
    }
    for (int i = start; i > stop; i += step) {
      idx++;
    }
  }
  return idx;
}
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_SEQUENCE_SLICE_H_
