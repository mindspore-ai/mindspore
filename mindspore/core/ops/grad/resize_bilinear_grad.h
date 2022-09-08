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

#ifndef MINDSPORE_CORE_OPS_GRAD_RESIZE_BILINEAR_GRAD_H_
#define MINDSPORE_CORE_OPS_GRAD_RESIZE_BILINEAR_GRAD_H_
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameResizeBilinearGrad = "ResizeBilinearGrad";
constexpr auto kGradIndex = 0;
constexpr auto kOriginalImageIndex = 1;
/// \brief Resizes an image to a certain size using the bilinear interpolation.
/// Refer to Python API @ref mindspore.ops.ResizeBilinearGrad for more details.
class MIND_API ResizeBilinearGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ResizeBilinearGrad);
  /// \brief Constructor.
  ResizeBilinearGrad() : BaseOperator(kNameResizeBilinearGrad) { InitIOName({"grads", "original_image"}, {"y"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.ResizeBilinearGrad for the inputs.
  void Init(const bool align_corners = false, const bool half_pixel_centers = false);
  /// \brief Set align_corners.
  void set_align_corners(const bool align_corners);
  /// \brief Set half_pixel_centers.
  void set_half_pixel_centers(const bool half_pixel_centers);
  /// \brief Get align_corners.
  ///
  /// \return align_corners.
  bool get_align_corners() const;
  /// \brief Get half_pixel_centers.
  ///
  /// \return half_pixel_centers.
  bool get_half_pixel_centers() const;
};
abstract::AbstractBasePtr ResizeBilinearGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_GRAD_RESIZE_BILINEAR_GRAD_H_
