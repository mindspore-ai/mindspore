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

#ifndef MINDSPORE_CORE_OPS_RESIZE_BICUBIC_H_
#define MINDSPORE_CORE_OPS_RESIZE_BICUBIC_H_
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameResizeBicubic = "ResizeBicubic";
class MIND_API ResizeBicubic : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ResizeBicubic);
  ResizeBicubic() : BaseOperator(kNameResizeBicubic) { InitIOName({"images", "size"}, {"y"}); }

  void Init(const bool align_corners = false, const bool half_pixel_centers = false);
  void set_align_corners(const bool align_corners);
  void set_half_pixel_centers(const bool half_pixel_centers);
  bool get_align_corners() const;
  bool get_half_pixel_centers() const;
  abstract::AbstractBasePtr ResizeBicubicInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args);
  using PrimResizeBicubic = std::shared_ptr<ResizeBicubic>;
};
abstract::AbstractBasePtr ResizeBicubicInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RESIZE_BICUBIC_H_
