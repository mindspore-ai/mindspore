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
#ifndef MINDSPORE_CORE_OPS_RESIZE_NEAREST_NEIGHBOR_V2_GRAD_H_
#define MINDSPORE_CORE_OPS_RESIZE_NEAREST_NEIGHBOR_V2_GRAD_H_

#include <string>
#include <memory>
#include <vector>

#include "abstract/abstract_value.h"
#include "ops/base_operator.h"
#include "utils/check_convert_utils.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameResizeNearestNeighborV2Grad = "ResizeNearestNeighborV2Grad";
/// \brief the grad operation of @ref mindspore.ops.ResizeNearestNeighborV2
/// Refer to Python API @ref mindspore._grad_ops.ResizeNearestNeighborV2Grad for more details.
class MIND_API ResizeNearestNeighborV2Grad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ResizeNearestNeighborV2Grad);

  /// \brief Constructor.
  ResizeNearestNeighborV2Grad() : BaseOperator(kNameResizeNearestNeighborV2Grad) {
    InitIOName({"grads", "size"}, {"y"});
  }

  bool get_align_corners() const;
  bool get_half_pixel_centers() const;
  std::string get_data_format() const;
};

AbstractBasePtr ResizeNearestNeighborV2GradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args);
using PrimResizeNearestNeighborV2GradPtr = std::shared_ptr<ResizeNearestNeighborV2Grad>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RESIZE_NEAREST_NEIGHBOR_V2_GRAD_H_
