/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_STRIDED_SLICE_V2_GRAD_H_
#define MINDSPORE_CORE_OPS_STRIDED_SLICE_V2_GRAD_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameStridedSliceV2Grad = "StridedSliceV2Grad";

class MIND_API StridedSliceV2Grad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(StridedSliceV2Grad);
  StridedSliceV2Grad() : BaseOperator(kNameStridedSliceV2Grad) {
    InitIOName({"shapex", "begin", "end", "strides", "dy"}, {"output"});
  }

  void Init(int64_t begin_mask = 0, int64_t end_mask = 0, int64_t ellipsis_mask = 0, int64_t new_axis_mask = 0,
            int64_t shrink_axis_mask = 0);
  void set_begin_mask(int64_t begin_mask);
  void set_end_mask(int64_t end_mask);
  void set_ellipsis_mask(int64_t ellipsis_mask);
  void set_new_axis_mask(int64_t new_axis_mask);
  void set_shrink_axis_mask(int64_t shrink_axis_mask);
  int64_t get_begin_mask() const;
  int64_t get_end_mask() const;
  int64_t get_ellipsis_mask() const;
  int64_t get_new_axis_mask() const;
  int64_t get_shrink_axis_mask() const;
};

abstract::AbstractBasePtr StridedSliceV2GradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimStridedSliceV2GradPtr = std::shared_ptr<StridedSliceV2Grad>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_STRIDED_SLICE_V2_GRAD_H_
