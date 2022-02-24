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

namespace mindspore {
namespace ops {
void Squeeze::Init(const std::vector<int64_t> &axis) { set_axis(axis); }
void Squeeze::set_axis(const std::vector<int64_t> &axis) { (void)AddAttr(kAxis, MakeValue(axis)); }
std::vector<int64_t> Squeeze::get_axis() const { return GetValue<std::vector<int64_t>>(GetAttr(kAxis)); }

namespace {
abstract::ShapePtr SqueezeInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto axis = GetValue<std::vector<int64_t>>(primitive->GetAttr(kAxis));
  std::vector<int64_t> ret_shape;
  std::vector<int64_t> ret_min_shape;
  std::vector<int64_t> ret_max_shape;

  auto shape_infos = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto in_shape = shape_infos[kShape];
  auto max_shape = shape_infos[kMaxShape];
  auto min_shape = shape_infos[kMinShape];

  auto len = SizeToLong(in_shape.size());
  if (axis.empty()) {
    for (int64_t i = 0; i < len; i++) {
      if (in_shape[i] != 1) {
        ret_shape.push_back(in_shape[LongToSize(i)]);
        if (!min_shape.empty() && !max_shape.empty()) {
          ret_min_shape.push_back(min_shape[LongToSize(i)]);
          ret_max_shape.push_back(max_shape[LongToSize(i)]);
        }
      }
    }
  } else {
    for (auto &item : axis) {
      CheckAndConvertUtils::CheckInRange<int64_t>("axis_or_elememt", item, kIncludeBoth, {-len, len + 1}, op_name);
      auto idx = item >= 0 ? item : len + item;
      if (in_shape[LongToSize(idx)] != 1L) {
        MS_EXCEPTION(ValueError) << "Cannot select an axis to squeeze out which has size not equal to one.";
      }
    }
    for (int64_t i = 0; i < len; i++) {
      auto it = std::find(axis.begin(), axis.end(), i);
      auto it2 = std::find(axis.begin(), axis.end(), i - len);
      if (!(it != axis.end() || it2 != axis.end())) {
        ret_shape.push_back(in_shape[LongToSize(i)]);
        if (!min_shape.empty() && !max_shape.empty()) {
          ret_min_shape.push_back(min_shape[LongToSize(i)]);
          ret_max_shape.push_back(max_shape[LongToSize(i)]);
        }
      }
    }
  }
  return std::make_shared<abstract::Shape>(ret_shape, ret_min_shape, ret_max_shape);
}

TypePtr SqueezeInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr arg) { return arg == nullptr; })) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  auto name = prim->name();
  MS_LOG(DEBUG) << "Infer data type for " << name;
  return input_args[0]->BuildType();
}
}  // namespace
AbstractBasePtr SqueezeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  const size_t x_index = 0;
  auto x_type = input_args[x_index]->BuildType();
  if (!x_type->isa<TensorType>()) {
    MS_EXCEPTION(TypeError) << "For Squeeze, the " << x_index << "'s input should be a Tensor, but got "
                            << x_type->ToString();
  }

  return abstract::MakeAbstract(SqueezeInferShape(primitive, input_args), SqueezeInferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(Squeeze, prim::kPrimSqueeze, SqueezeInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
