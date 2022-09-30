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

#include "ops/clip_by_norm.h"
#include <set>
#include <map>
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
void CheckAxisValid(const PrimitivePtr &primitive, const std::vector<int64_t> &x_shape) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto axis_value = primitive->GetAttr(kAxis);
  MS_EXCEPTION_IF_NULL(axis_value);
  std::vector<int64_t> axis;
  if (axis_value->isa<ValueSequence>()) {
    axis = GetValue<std::vector<int64_t>>(axis_value);
  } else if (axis_value->isa<Int64Imm>()) {
    axis.emplace_back(GetValue<int64_t>(axis_value));
  } else {
    MS_EXCEPTION(TypeError) << "For `" << kNameClipByNorm << "`, the type of attribute `axis` is invalid.";
  }
  // Check whether the value range of the axis is reasonable
  const auto x_dim = SizeToLong(x_shape.size());
  if (!axis.empty()) {
    bool invalid = std::any_of(axis.begin(), axis.end(),
                               [&x_dim](const int64_t &value) { return value >= x_dim || value < -x_dim; });
    if (invalid) {
      MS_EXCEPTION(ValueError) << "The value in attribute `axis` should be within [" << -x_dim << ", " << x_dim
                               << ") range.";
    }
  }
}

abstract::ShapePtr ClipByNormInferShape(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args_abs) {
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args_abs[0]->GetShapeTrack());
  auto clip_norm_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args_abs[1]->GetShapeTrack());
  auto x_shape = x_shape_map.at(kShape);
  auto clip_norm_shape = clip_norm_shape_map.at(kShape);
  // Check whether dynamic shape exists.
  bool is_x_dyn = std::any_of(x_shape.begin(), x_shape.end(),
                              [](const int64_t &value) { return value == abstract::Shape::kShapeDimAny; });
  bool is_clip_norm_dyn = std::any_of(clip_norm_shape.begin(), clip_norm_shape.end(),
                                      [](const int64_t &value) { return value == abstract::Shape::kShapeDimAny; });
  if (is_x_dyn || is_clip_norm_dyn) {
    MS_EXCEPTION(ValueError) << "For `" << kNameClipByNorm
                             << "` op, dynamic shape is not supported now, but got `-1` in input args shape.";
  }
  // Check whether clip_norm shape is valid
  if (clip_norm_shape != x_shape) {
    const auto broadcast_shape = CalBroadCastShape(x_shape, clip_norm_shape, kNameClipByNorm, "input_x", "clip_norm");
    bool clip_norm_shape_all_ones =
      std::all_of(clip_norm_shape.begin(), clip_norm_shape.end(), [](const int64_t &v) { return v == 1; });
    if (broadcast_shape != x_shape && !clip_norm_shape_all_ones) {
      MS_EXCEPTION(ValueError) << "The shape of `clip_norm` only support `()`、`(1)` or a shape can be broadcast to "
                                  "input `x` shape, but got input `x` shape: "
                               << x_shape << ", `clip_norm` shape: " << clip_norm_shape;
    }
  }
  // Check whether the attribute `axis` is valid.
  CheckAxisValid(primitive, x_shape);
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr ClipByNormInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args_abs) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x_type = CheckAndConvertUtils::GetTensorInputType(kNameClipByNorm, input_args_abs, 0);
  auto clip_norm_type = CheckAndConvertUtils::GetTensorInputType(kNameClipByNorm, input_args_abs, 1);
  const std::set<TypePtr> supported_types{kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckSubClass("x", x_type, supported_types, kNameClipByNorm);
  (void)CheckAndConvertUtils::CheckSubClass("clip_norm", clip_norm_type, supported_types, kNameClipByNorm);
  return kFloat32;
}
}  // namespace

void ClipByNorm::Init(const int64_t axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }

void ClipByNorm::Init(const std::vector<int64_t> &axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }

std::vector<int64_t> ClipByNorm::GetAxis() const {
  std::vector<int64_t> axis;
  auto axis_value = GetAttr(kAxis);
  MS_EXCEPTION_IF_NULL(axis_value);
  if (axis_value->isa<api::ValueSequence>()) {
    axis = api::GetValue<std::vector<int64_t>>(axis_value);
  } else if (axis_value->isa<api::Int64Imm>()) {
    axis.emplace_back(api::GetValue<int64_t>(axis_value));
  } else {
    MS_EXCEPTION(TypeError) << "For `" << kNameClipByNorm << "`, the type of attribute `axis` is invalid.";
  }
  return axis;
}

MIND_API_OPERATOR_IMPL(ClipByNorm, BaseOperator);
AbstractBasePtr ClipByNormInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args_abs) {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_CHECK_FAIL(primitive->name() == kNameClipByNorm, "The op name is not ClipByNorm.");
  const int64_t kInputNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args_abs, kEqual, kInputNum, kNameClipByNorm);
  MS_EXCEPTION_IF_NULL(input_args_abs[0]);
  MS_EXCEPTION_IF_NULL(input_args_abs[1]);
  if (!input_args_abs[0]->isa<abstract::AbstractTensor>() || !input_args_abs[1]->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "For '" << kNameClipByNorm << "', the input args must be tensor.";
  }
  auto infer_type = ClipByNormInferType(primitive, input_args_abs);
  auto infer_shape = ClipByNormInferShape(primitive, input_args_abs);
  auto abs = abstract::MakeAbstract(infer_shape, infer_type);
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractScalar>()) {
    return std::make_shared<abstract::AbstractTensor>(abs);
  }
  return abs;
}
REGISTER_PRIMITIVE_EVAL_IMPL(ClipByNorm, prim::kPrimClipByNorm, ClipByNormInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
