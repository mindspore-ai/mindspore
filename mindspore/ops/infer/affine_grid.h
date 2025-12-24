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

#ifndef MINDSPORE_CORE_OPS_AFFINE_GRID_H_
#define MINDSPORE_CORE_OPS_AFFINE_GRID_H_
#include <memory>
#include <vector>
#include "abstract/abstract_value.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAffineGrid = "AffineGrid";
/// \brief Returns a Tensor whose value is evenly spaced in the interval theta and output_size (including
/// align_corners). Refer to Python API @ref mindspore.ops.AffineGrid for more details.
class OPS_API AffineGrid : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(AffineGrid);
  /// \brief Constructor.
  AffineGrid() : BaseOperator(kNameAffineGrid) { InitIOName({"theta", "output_size"}, {"y"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.FloorDiv for the inputs.
  void Init(const bool align_corners = false);
  /// \brief Set align_corners. Defaults to false.
  void set_align_corners(const bool align_corners);
  /// \brief Get align_corners.
  ///
  /// \return align_corners.
  bool get_align_corners() const;
};
OPS_API abstract::AbstractBasePtr AffineGridInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimAffineGridPtr = std::shared_ptr<AffineGrid>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_AFFINE_GRID_H_
