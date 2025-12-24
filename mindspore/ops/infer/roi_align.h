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

#ifndef MINDSPORE_CORE_OPS_ROI_ALIGN_H_
#define MINDSPORE_CORE_OPS_ROI_ALIGN_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameROIAlign = "ROIAlign";
/// \brief ROIAlign defined the ROIAlign operator prototype.
class OPS_API ROIAlign : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ROIAlign);
  /// \brief Constructor.
  ROIAlign() : BaseOperator(kNameROIAlign) { InitIOName({"feature", "rois"}, {"output"}); }
  void Init(const int64_t pooled_height, const int64_t pooled_weight, const float spatial_scale,
            const int64_t sample_num = 2, const int64_t roi_end_mode = 1);

  /// \brief Method to set pooled_height attribute.
  ///
  /// \param[in] pooled_height Define the height of the output.
  void set_pooled_height(const int64_t pooled_height);

  /// \brief Method to get pooled_height attribute.
  ///
  /// \return the height of the output.
  int64_t get_pooled_height() const;

  /// \brief Method to set pooled_width attribute.
  ///
  /// \param[in] pooled_width Define the width of the output.
  void set_pooled_width(const int64_t pooled_width);

  /// \brief Method to get pooled_width attribute.
  ///
  /// \return the height of the output.
  int64_t get_pooled_width() const;

  /// \brief Method to set spatial_scale attribute.
  ///
  /// \param[in] spatial_scale Define spatial_scale.
  void set_spatial_scale(const float spatial_scale);

  /// \brief Method to get spatial_scale attribute.
  ///
  /// \return the spatial_scale.
  float get_spatial_scale() const;

  /// \brief Method to set sample num attribute.
  ///
  /// \param[in] sample_num Define sample_num.
  void set_sample_num(const int64_t sample_num);

  /// \brief Method to get sample_num attribute.
  ///
  /// \return the sample_num.
  int64_t get_sample_num() const;

  /// \brief Method to set roi_end_mode attribute.
  ///
  /// \param[in] sample_num Define roi_end_mode.
  void set_roi_end_mode(const int64_t roi_end_mode);

  /// \brief Method to get roi_end_mode attribute.
  ///
  /// \return the roi_end_mode.
  int64_t get_roi_end_mode() const;
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_ROI_ALIGN_H_
