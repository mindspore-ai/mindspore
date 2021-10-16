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

#ifndef MINDSPORE_CORE_OPS_CROP_AND_RESIZE_H_
#define MINDSPORE_CORE_OPS_CROP_AND_RESIZE_H_
#include <vector>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameCropAndResize = "CropAndResize";
/// \brief Extracts crops from the input image tensor and resizes them.
/// Refer to Python API @ref mindspore.ops.CropAndResize for more details.
class MS_CORE_API CropAndResize : public PrimitiveC {
 public:
  /// \brief Constructor.
  CropAndResize() : PrimitiveC(kNameCropAndResize) { InitIOName({"x", "boxes", "box_index", "crop_size"}, {"y"}); }
  /// \brief Destructor.
  ~CropAndResize() = default;
  MS_DECLARE_PARENT(CropAndResize, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.CropAndResize for the inputs.
  void Init(ResizeMethod method, float extrapolation_value);

  /// \brief Set method.
  void set_method(ResizeMethod method);
  /// \brief Set extrapolation_value.
  void set_extrapolation_value(float extrapolation_value);
  /// \brief Get method.
  ///
  /// \return method.
  ResizeMethod get_method() const;
  /// \brief Get extrapolation_value.
  ///
  /// \return extrapolation_value.
  float get_extrapolation_value() const;
};
using PrimCropAndResizePtr = std::shared_ptr<CropAndResize>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_CROP_AND_RESIZE_H_
