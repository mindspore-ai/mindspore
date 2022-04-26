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

#ifndef MINDSPORE_CORE_OPS_GRID_SAMPLER_2D_H_
#define MINDSPORE_CORE_OPS_GRID_SAMPLER_2D_H_
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameGridSampler2D = "GridSampler2D";
class MIND_API GridSampler2D : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(GridSampler2D);
  GridSampler2D() : BaseOperator(kNameGridSampler2D) { InitIOName({"input_x", "grid"}, {"output"}); }
  std::string get_interpolation_mode() const;
  std::string get_padding_mode() const;
  bool get_align_corners() const;
};
abstract::AbstractBasePtr GridSampler2DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimGridSampler2D = std::shared_ptr<GridSampler2D>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_GRID_SAMPLER_2D_H_
