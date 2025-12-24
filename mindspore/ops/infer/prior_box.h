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

#ifndef MINDSPORE_CORE_OPS_PRIOR_BOX_H_
#define MINDSPORE_CORE_OPS_PRIOR_BOX_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNamePriorBox = "PriorBox";
/// \brief PriorBox defined PriorBox operator prototype of lite.
class OPS_API PriorBox : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(PriorBox);
  /// \brief Constructor.
  PriorBox() : BaseOperator(kNamePriorBox) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] min_sizes Define the minimum side length of square boxes, can be multiple.
  /// \param[in] max_sizes Define the maximum side length of square boxes as sqrt(min_size * max_size), can be multiple.
  /// \param[in] aspect_ratios Define the aspect ratios of generated boxes. For each aspect_ratio, the width and height
  ///            are min_size * sqrt(aspect_ratio) and min_size / sqrt(aspect_ratio) respectively.
  /// \param[in] variances Define variances for adjusting the prior boxes.
  /// \param[in] image_size_w Define the width of the image.
  /// \param[in] image_size_h Define the height of the image.
  /// \param[in] step_w Define the ratio of the image width and the feature map width.
  /// \param[in] step_h Define the ratio of the image height and the feature map height.
  /// \param[in] clip Define whether clip the prior boxes to [0.0, 1.0].
  /// \param[in] flip Define whether flip aspect ratios. If true, with an aspect ratio r, ratio 1.0/r will be generated.
  /// \param[in] offset Define the offset to the zero points of width and height.
  void Init(const std::vector<int64_t> &min_sizes, const std::vector<int64_t> &max_sizes,
            const std::vector<float> &aspect_ratios, const std::vector<float> &variances, const int64_t image_size_w,
            const int64_t image_size_h, const float step_w, const float step_h, const bool clip, const bool flip,
            const float offset);

  /// \brief Method to set min_sizes attribute.
  ///
  /// \param[in] min_sizes Define the minimum side length of square boxes, can be multiple.
  void set_min_sizes(const std::vector<int64_t> &min_sizes);

  /// \brief Method to set max_sizes attribute.
  ///
  /// \param[in] max_sizes Define the maximum side length of square boxes as sqrt(min_size * max_size), can be multiple.
  void set_max_sizes(const std::vector<int64_t> &max_sizes);

  /// \brief Method to set aspect_ratios attribute.
  ///
  /// \param[in] aspect_ratios Define the aspect ratios of generated boxes. For each aspect_ratio, the width and height
  ///            are min_size * sqrt(aspect_ratio) and min_size / sqrt(aspect_ratio) respectively.
  void set_aspect_ratios(const std::vector<float> &aspect_ratios);

  /// \brief Method to set variances attribute.
  ///
  /// \param[in] variances Define variances for adjusting the prior boxes.
  void set_variances(const std::vector<float> &variances);

  /// \brief Method to set image_size_w attribute.
  ///
  /// \param[in] image_size_w Define the width of the image.
  void set_image_size_w(const int64_t image_size_w);

  /// \brief Method to set image_size_h attribute.
  ///
  /// \param[in] image_size_h Define the height of the image.
  void set_image_size_h(const int64_t image_size_h);

  /// \brief Method to set step_w attribute.
  ///
  /// \param[in] step_w Define the ratio of the image width and the feature map width.
  void set_step_w(const float step_w);

  /// \brief Method to set step_h attribute.
  ///
  /// \param[in] step_h Define the ratio of the image height and the feature map height.
  void set_step_h(const float step_h);

  /// \brief Method to set clip attribute.
  ///
  /// \param[in] clip Define whether clip the prior boxes to [0.0, 1.0].
  void set_clip(const bool clip);

  /// \brief Method to set flip attribute.
  ///
  /// \param[in] flip Define whether flip aspect ratios. If true, with an aspect ratio r, ratio 1.0/r will be generated.
  void set_flip(const bool flip);

  /// \brief Method to set offset attribute.
  ///
  /// \param[in] offset Define the offset to the zero points of width and height.
  void set_offset(const float offset);

  /// \brief Method to get min_sizes attribute.
  ///
  /// \return min_sizes attribute.
  std::vector<int64_t> get_min_sizes() const;

  /// \brief Method to get max_sizes attribute.
  ///
  /// \return max_sizes attribute.
  std::vector<int64_t> get_max_sizes() const;

  /// \brief Method to get aspect_ratios attribute.
  ///
  /// \return aspect_ratios attribute.
  std::vector<float> get_aspect_ratios() const;

  /// \brief Method to get variances attribute.
  ///
  /// \return variances attribute.
  std::vector<float> get_variances() const;

  /// \brief Method to get image_size_w attribute.
  ///
  /// \return image_size_w attribute.
  int64_t get_image_size_w() const;

  /// \brief Method to get image_size_h attribute.
  ///
  /// \return image_size_h attribute.
  int64_t get_image_size_h() const;

  /// \brief Method to get step_w attribute.
  ///
  /// \return step_w attribute.
  float get_step_w() const;

  /// \brief Method to get step_h attribute.
  ///
  /// \return step_h attribute.
  float get_step_h() const;

  /// \brief Method to get flip attribute.
  ///
  /// \return flip attribute.
  bool get_flip() const;

  /// \brief Method to get clip attribute.
  ///
  /// \return clip attribute.
  bool get_clip() const;

  /// \brief Method to get offset attribute.
  ///
  /// \return offset attribute.
  float get_offset() const;
};

OPS_API abstract::AbstractBasePtr PriorBoxInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_PRIOR_BOX_H_
