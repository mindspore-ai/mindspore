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

#ifndef MINDSPORE_CORE_OPS_COMBINED_NON_MAX_SUPPRESSION_H_
#define MINDSPORE_CORE_OPS_COMBINED_NON_MAX_SUPPRESSION_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameCombinedNonMaxSuppression = "CombinedNonMaxSuppression";
/// \brief Greedily selects a subset of bounding boxes in descending order of score.
/// Refer to Python API @ref mindspore.ops.CombineNonMaxSuppression for more details.
class OPS_API CombinedNonMaxSuppression : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(CombinedNonMaxSuppression);
  /// \brief Constructor.
  CombinedNonMaxSuppression() : BaseOperator(kNameCombinedNonMaxSuppression) {
    InitIOName({"boxes", "scores", "max_output_size_per_class", "max_total_size", "iou_threshold", "score_threshold"},
               {"nmsed_box", "nmsed_scores", "nmsed_classes", "valid_detections"});
  }
  bool get_pad_per_class() const;
  bool get_clip_boxes() const;
};
OPS_API abstract::AbstractBasePtr CombinedNonMaxSuppressionInfer(
  const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
  const std::vector<abstract::AbstractBasePtr> &input_args);

using kPrimCombinedNonMaxSuppressionPtr = std::shared_ptr<CombinedNonMaxSuppression>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_COMBINED_NON_MAX_SUPPRESSION_H_
