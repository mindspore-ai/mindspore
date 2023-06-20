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

#ifndef MINDSPORE_CORE_OPS_GRAD_GRID_SAMPLER_3D_GRAD_H_
#define MINDSPORE_CORE_OPS_GRAD_GRID_SAMPLER_3D_GRAD_H_
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameGridSampler3DGrad = "GridSampler3DGrad";
class MIND_API GridSampler3DGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(GridSampler3DGrad);
  GridSampler3DGrad() : BaseOperator(kNameGridSampler3DGrad) {
    InitIOName({"grad", "input_x", "grid"}, {"dx", "dgrid"});
  }
  std::string get_interpolation_mode() const;
  std::string get_padding_mode() const;
  bool get_align_corners() const;
};
MIND_API abstract::AbstractBasePtr GridSampler3DGradInfer(const abstract::AnalysisEnginePtr &,
                                                          const PrimitivePtr &primitive,
                                                          const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimGridSampler3DGrad = std::shared_ptr<GridSampler3DGrad>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_GRAD_GRID_SAMPLER_3D_GRAD_H_
