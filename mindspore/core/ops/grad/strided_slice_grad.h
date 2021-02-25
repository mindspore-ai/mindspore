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

#ifndef MINDSPORE_CORE_OPS_STRIDED_SLICE_GRAD_H_
#define MINDSPORE_CORE_OPS_STRIDED_SLICE_GRAD_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameStridedSliceGrad = "StridedSliceGrad";
class StridedSliceGrad : public PrimitiveC {
 public:
  StridedSliceGrad() : PrimitiveC(kNameStridedSliceGrad) {}
  ~StridedSliceGrad() = default;
  MS_DECLARE_PARENT(StridedSliceGrad, PrimitiveC);
  void Init(const int64_t begin_mask = 0, const int64_t end_mask = 0, const int64_t ellipsis_mask = 0,
            const int64_t new_axis_mask = 0, const int64_t shrink_axis_mask = 0);
  void set_begin_mask(const int64_t begin_mask);
  void set_end_mask(const int64_t end_mask);
  void set_ellipsis_mask(const int64_t ellipsis_mask);
  void set_new_axis_mask(const int64_t new_axis_mask);
  void set_shrink_axis_mask(const int64_t shrink_axis_mask);
  int64_t get_begin_mask() const;
  int64_t get_end_mask() const;
  int64_t get_ellipsis_mask() const;
  int64_t get_new_axis_mask() const;
  int64_t get_shrink_axis_mask() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_STRIDED_SLICE_GRAD_H_
