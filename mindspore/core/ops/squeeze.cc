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

#include "ops/squeeze.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void Squeeze::Init(const std::vector<int64_t> &axis) { set_axis(axis); }
void Squeeze::set_axis(const std::vector<int64_t> &axis) { (void)AddAttr(kAxis, api::MakeValue(axis)); }
std::vector<int64_t> Squeeze::get_axis() const { return GetValue<std::vector<int64_t>>(GetAttr(kAxis)); }
namespace {
constexpr auto kSqueezedDim = 1;
abstract::ShapePtr SqueezeInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto axis = GetValue<std::vector<int64_t>>(primitive->GetAttr(kAxis));
  std::vector<int64_t> ret_shape;

  auto shape_infos = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto in_shape = shape_infos[kShape];

  if (IsDynamicRank(in_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }

  if (axis.empty()) {
    for (size_t i = 0; i < in_shape.size(); i++) {
      if (in_shape[i] != kSqueezedDim) {
        ret_shape.push_back(in_shape[i]);
      }
    }
  } else {
    auto rank = SizeToLong(in_shape.size());
    for (auto &item : axis) {
      CheckAndConvertUtils::CheckInRange<int64_t>("element or value of axis", item, kIncludeLeft, {-rank, rank},
                                                  op_name);
      auto idx = item >= 0 ? item : rank + item;
      // If shape dims contain unknown dim, ignore it.
      if (in_shape[LongToSize(idx)] != abstract::Shape::kShapeDimAny) {
        const std::string ith_shape = "input_x.shape[" + std::to_string(idx) + "]";
        (void)CheckAndConvertUtils::CheckValue<int64_t>(ith_shape, in_shape[LongToSize(idx)], kEqual, kSqueezedDim,
                                                        op_name);
      }
    }
    for (int64_t i = 0; i < rank; i++) {
      auto it = std::find(axis.begin(), axis.end(), i);
      auto it2 = std::find(axis.begin(), axis.end(), i - rank);
      if (!(it != axis.end() || it2 != axis.end())) {
        ret_shape.push_back(in_shape[LongToSize(i)]);
      }
    }
  }
  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr SqueezeInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const auto prim_name = prim->name();
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
  return input_args[kInputIndex0]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(Squeeze, BaseOperator);
AbstractBasePtr SqueezeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  constexpr int64_t squeeze_input_length = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, squeeze_input_length, primitive->name());
  auto type = SqueezeInferType(primitive, input_args);
  auto shape = SqueezeInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Squeeze, prim::kPrimSqueeze, SqueezeInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
