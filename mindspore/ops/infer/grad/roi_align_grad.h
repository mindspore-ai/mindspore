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

#ifndef MINDSPORE_CORE_OPS_ROI_ALIGN_GRAD_H_
#define MINDSPORE_CORE_OPS_ROI_ALIGN_GRAD_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameROIAlignGrad = "ROIAlignGrad";

class OPS_API ROIAlignGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ROIAlignGrad);
  ROIAlignGrad() : BaseOperator(kNameROIAlignGrad) { InitIOName({"dy", "rois", "xdiff_shape"}, {"dx"}); }

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] pooled_height Define the height of the output.
  /// \param[in] pooled_width Define the width of the output.
  /// \param[in] spatial_scale Define the size factor to translate ROI coordinates from the input coordinate to the
  /// actual coordinate.
  /// \param[in] sample_num Define the number of sampling points.
  void Init(const int64_t pooled_height, const int64_t pooled_width, const float spatial_scale,
            const int64_t sample_num);

  /// \brief Method to set pooled_height attribute.
  ///
  /// \param[in] pooled_height Define the height of the output.
  void set_pooled_height(const int64_t pooled_height);

  /// \brief Method to set pooled_width attribute.
  ///
  /// \param[in] pooled_width Define the width of the output.
  void set_pooled_width(const int64_t pooled_width);

  /// \brief Method to set spatial_scale attribute.
  ///
  /// \param[in] spatial_scale Define the size factor to translate ROI coordinates from the input coordinate to the
  /// actual
  ///            coordinate.
  void set_spatial_scale(const float spatial_scale);

  /// \brief Method to set sample_num attribute.
  ///
  /// \param[in] sample_num Define the number of sampling points.
  void set_sample_num(const int64_t sample_num);

  /// \brief Method to get pooled_height attribute.
  ///
  /// \return the height of the output.
  int64_t get_pooled_height() const;

  /// \brief Method to get pooled_width attribute.
  ///
  /// \return the width of the output.
  int64_t get_pooled_width() const;

  /// \brief Method to get spatial_scale attribute.
  ///
  /// \return the size factor.
  float get_spatial_scale() const;

  /// \brief Method to get sample_num attribute.
  ///
  /// \return the number of sampling points.
  int64_t get_sample_num() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ROI_ALIGN_GRAD_H_
