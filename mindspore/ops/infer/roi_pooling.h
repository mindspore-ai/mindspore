/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_ROI_POOLING_H_
#define MINDSPORE_CORE_OPS_ROI_POOLING_H_
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameROIPooling = "ROIPooling";
/// \brief ROIPooling defined the ROIPooling operator prototype.
class OPS_API ROIPooling : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ROIPooling);
  /// \brief Constructor.
  ROIPooling() : BaseOperator(kNameROIPooling) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] pooled_h Define the height of the output.
  /// \param[in] pooled_w Define the width of the output.
  /// \param[in] scale Define the size factor to translate ROI coordinates from the input coordinate to the actual
  ///            coordinate.
  void Init(const int64_t pooled_h, const int64_t pooled_w, const float scale);

  /// \brief Method to set pooled_h attribute.
  ///
  /// \param[in] pooled_h Define the height of the output.
  void set_pooled_h(const int64_t pooled_h);

  /// \brief Method to set pooled_w attribute.
  ///
  /// \param[in] pooled_w Define the width of the output.
  void set_pooled_w(const int64_t pooled_w);

  /// \brief Method to set scale attribute.
  ///
  /// \param[in] scale Define the size factor to translate ROI coordinates from the input coordinate to the actual
  ///            coordinate.
  void set_scale(const float scale);

  /// \brief Method to get pooled_h attribute.
  ///
  /// \return the height of the output.
  int64_t get_pooled_h() const;

  /// \brief Method to get pooled_w attribute.
  ///
  /// \return the width of the output.
  int64_t get_pooled_w() const;

  /// \brief Method to get scale attribute.
  ///
  /// \return the size factor.
  float get_scale() const;
};
OPS_API abstract::AbstractBasePtr ROIPoolingInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ROI_POOLING_H_
