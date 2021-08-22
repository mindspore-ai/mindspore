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
#include <vector>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameResize = "Resize";
class Resize : public PrimitiveC {
 public:
  Resize() : PrimitiveC(kNameResize) {}
  ~Resize() = default;
  MS_DECLARE_PARENT(Resize, PrimitiveC);
  void Init(const Format format, const ResizeMethod method, const int64_t new_height, const int64_t new_width,
            const bool preserve_aspect_ratio, const CoordinateTransformMode coordinate_transform_mode,
            const float cubic_coeff, const int64_t exclude_outside, const float extrapolation_value,
            const NearestMode nearest_mode);
  void set_format(const Format format);
  void set_method(const ResizeMethod method);
  void set_new_height(const int64_t new_height);
  void set_new_width(const int64_t new_width);
  void set_preserve_aspect_ratio(const bool preserve_aspect_ratio);
  void set_coordinate_transform_mode(const CoordinateTransformMode coordinate_transform_mode);
  void set_cubic_coeff(const float cubic_coeff);
  void set_exclude_outside(const int64_t exclude_outside);
  void set_extrapolation_value(const float extrapolation_value);
  void set_nearest_mode(const NearestMode nearest_mode);
  Format get_format() const;
  ResizeMethod get_method() const;
  int64_t get_new_height() const;
  int64_t get_new_width() const;
  bool get_preserve_aspect_ratio() const;
  CoordinateTransformMode get_coordinate_transform_mode() const;
  float get_cubic_coeff() const;
  int64_t get_exclude_outside() const;
  float get_extrapolation_value() const;
  NearestMode get_nearest_mode() const;
};

AbstractBasePtr ResizeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args);
using PrimResizePtr = std::shared_ptr<Resize>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RESIZE_H_
