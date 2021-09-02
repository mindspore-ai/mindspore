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
#include "ops/resize_bilinear.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
void ResizeBilinear::set_size(const std::vector<int64_t> &size) { (void)this->AddAttr(kSize, MakeValue(size)); }

std::vector<int64_t> ResizeBilinear::get_size() const { return GetValue<std::vector<int64_t>>(GetAttr(kSize)); }

void ResizeBilinear::set_align_corners(const bool align_corners) {
  (void)this->AddAttr(kAlignCorners, MakeValue(align_corners));
}

bool ResizeBilinear::get_align_corners() const {
  auto value_ptr = GetAttr(kAlignCorners);
  return GetValue<bool>(value_ptr);
}

void ResizeBilinear::Init(const std::vector<int64_t> &size, const bool align_corners) {
  this->set_size(size);
  this->set_align_corners(align_corners);
}
AbstractBasePtr ResizeBilinearInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("infer", SizeToLong(input_args.size()), kEqual, 1, prim_name);

  // Infer shape
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  const int64_t shape_size = 4;
  (void)CheckAndConvertUtils::CheckInteger("input rank", SizeToLong(input_shape.size()), kEqual, shape_size, prim_name);
  std::vector<int64_t> out_shape = {input_shape[0], input_shape[1]};
  auto size = GetValue<std::vector<int64_t>>(primitive->GetAttr(kSize));
  (void)out_shape.insert(out_shape.end(), size.begin(), size.end());

  // Infer type
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_type", input_args[0]->BuildType(), valid_types, prim_name);
  return std::make_shared<abstract::AbstractTensor>(input_args[0]->BuildType(), out_shape);
}
REGISTER_PRIMITIVE_C(kNameResizeBilinear, ResizeBilinear);
}  // namespace ops
}  // namespace mindspore
