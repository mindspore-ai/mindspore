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

#ifndef MINDSPORE_CORE_OPS_RESIZE_H_
#define MINDSPORE_CORE_OPS_RESIZE_H_
#include <memory>

#include "mindapi/base/format.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameResize = "Resize";
/// \brief Resize defined the Resize operator prototype of lite.
class OPS_API Resize : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Resize);
  /// \brief Constructor.
  Resize() : BaseOperator(kNameResize) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] format Define the format of the input, which only support NHWC on lite.
  /// \param[in] method Define the mode of resizing.
  /// \param[in] new_height Define the height of the output.
  /// \param[in] new_width Define the width of the output.
  /// \param[in] preserve_aspect_ratio Define a boolean to indicate keep the aspect radio with the input. Default is
  ///            false.
  /// \param[in] coordinate_transform_mode Define the rule to map coordinate.
  /// \param[in] cubic_coeff Define a coefficient only used in cubic interpolation.
  /// \param[in] exclude_outside Define a value to indicate whether to set the outside of the sampling area as 0. If the
  ///            value is 1, the outside area will be set as 0. Default is 0.
  /// \param[in] extrapolation_value Define a value that will be used to fill the outside of original area if possible.
  /// \param[in] nearest_mode Define the rule how to get nearest pixel.
  void Init(const Format format, const ResizeMethod method, const int64_t new_height, const int64_t new_width,
            const bool preserve_aspect_ratio, const CoordinateTransformMode coordinate_transform_mode,
            const float cubic_coeff, const int64_t exclude_outside, const float extrapolation_value,
            const NearestMode nearest_mode);

  /// \brief Method to set format attribute.
  ///
  /// \param[in] format Define the format of the input, which only support NHWC on lite.
  void set_format(const Format format);

  /// \brief Method to set method attribute.
  ///
  /// \param[in] method Define the mode of resizing.
  void set_method(const ResizeMethod method);

  /// \brief Method to set new_height attribute.
  ///
  /// \param[in] new_height Define the height of the output.
  void set_new_height(const int64_t new_height);

  /// \brief Method to set new_width attribute.
  ///
  /// \param[in] new_width Define the width of the output.
  void set_new_width(const int64_t new_width);

  /// \brief Method to set preserve_aspect_ratio attribute.
  ///
  /// \param[in] preserve_aspect_ratio Define a boolean to indicate keep the aspect radio with the input. Default is
  ///            false.
  void set_preserve_aspect_ratio(const bool preserve_aspect_ratio);

  /// \brief Method to set coordinate_transform_mode attribute.
  ///
  /// \param[in] coordinate_transform_mode Define the rule to map coordinate.
  void set_coordinate_transform_mode(const CoordinateTransformMode coordinate_transform_mode);

  /// \brief Method to set cubic_coeff attribute.
  ///
  /// \param[in] cubic_coeff Define a coefficient only used in cubic interpolation.
  void set_cubic_coeff(const float cubic_coeff);

  /// \brief Method to set exclude_outside attribute.
  ///
  /// \param[in] exclude_outside Define a value to indicate whether to set the outside of the sampling area as 0. If the
  ///            value is 1, the outside area will be set as 0. Default is 0.
  void set_exclude_outside(const int64_t exclude_outside);

  /// \brief Method to set extrapolation_value attribute.
  ///
  /// \param[in] extrapolation_value Define a value that will be used to fill the outside of original area if possible.
  void set_extrapolation_value(const float extrapolation_value);

  /// \brief Method to set nearest_mode attribute.
  ///
  /// \param[in] nearest_mode Define the rule how to get nearest pixel.
  void set_nearest_mode(const NearestMode nearest_mode);

  /// \brief Method to get format attribute.
  ///
  /// \return the format of the input.
  Format get_format() const;

  /// \brief Method to get method attribute.
  ///
  /// \return the mode of resizing.
  ResizeMethod get_method() const;

  /// \brief Method to get new_height attribute.
  ///
  /// \return the height of the output.
  int64_t get_new_height() const;

  /// \brief Method to get new_width attribute.
  ///
  /// \return the width of the output.
  int64_t get_new_width() const;

  /// \brief Method to get preserve_aspect_ratio attribute.
  ///
  /// \return a boolean value.
  bool get_preserve_aspect_ratio() const;

  /// \brief Method to get coordinate_transform_mode attribute.
  ///
  /// \return the rule to map coordinate
  CoordinateTransformMode get_coordinate_transform_mode() const;

  /// \brief Method to get cubic_coeff attribute.
  ///
  /// \return a coefficient used in cubic interpolation
  float get_cubic_coeff() const;

  /// \brief Method to get exclude_outside attribute.
  ///
  /// \return a value to indicate whether to set the outside of the sampling area as 0.
  int64_t get_exclude_outside() const;

  /// \brief Method to get extrapolation_value attribute.
  ///
  /// \return a value used to fill the outside of original area if possible
  float get_extrapolation_value() const;

  /// \brief Method to get nearest_mode attribute.
  ///
  /// \return  the rule to get nearest pixel.
  NearestMode get_nearest_mode() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RESIZE_H_
