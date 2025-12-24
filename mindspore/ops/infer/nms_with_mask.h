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

#ifndef MINDSPORE_CORE_OPS_NMS_WITH_MASK_H
#define MINDSPORE_CORE_OPS_NMS_WITH_MASK_H
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameNMSWithMask = "NMSWithMask";
/// \brief NMSWithMask operator. Refer to Python API @ref mindspore.ops.NMSWithMask for more details.
class OPS_API NMSWithMask : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(NMSWithMask);
  /// \brief Constructor.
  NMSWithMask() : BaseOperator(kNameNMSWithMask) {
    InitIOName({"bboxes"}, {"output_boxes", "output_idx", "selected_mask"});
  }
  explicit NMSWithMask(const std::string k_name) : BaseOperator(k_name) {
    InitIOName({"bboxes"}, {"output_boxes", "output_idx", "selected_mask"});
  }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.NMSWithMask for the inputs.
  void Init(const float iou_threshold = 0.5);
  /// \brief Set iou_threshold.
  void set_iou_threshold(const std::vector<float> &iou_thredshold);
  /// \brief Get iou_threshold.
  ///
  /// \return iou_threshold.
  std::vector<float> get_iou_threshold() const;
};

OPS_API abstract::AbstractBasePtr NMSWithMaskInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                   const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_NMS_WITH_MASK_H
