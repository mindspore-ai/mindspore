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
class CropAndResize : public PrimitiveC {
 public:
  CropAndResize() : PrimitiveC(kNameCropAndResize) { InitIOName({"x", "boxes", "box_index", "crop_size"}, {"y"}); }
  ~CropAndResize() = default;
  MS_DECLARE_PARENT(CropAndResize, PrimitiveC);
  void Init(const ResizeMethod method, const float extrapolation_value);

  void set_method(const ResizeMethod method);
  void set_extrapolation_value(const float extrapolation_value);

  ResizeMethod get_method() const;
  float get_extrapolation_value() const;
};

AbstractBasePtr CropAndResizeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args);
using PrimCropAndResizePtr = std::shared_ptr<CropAndResize>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_CROP_AND_RESIZE_H_
