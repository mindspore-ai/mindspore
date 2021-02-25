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
void Squeeze::set_axis(const std::vector<int64_t> &axis) { AddAttr(kAxis, MakeValue(axis)); }
std::vector<int64_t> Squeeze::get_axis() const {
  auto value_ptr = this->GetAttr(kAxis);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto squeeze_prim = primitive->cast<PrimSqueezePtr>();
  MS_EXCEPTION_IF_NULL(squeeze_prim);
  auto op_name = squeeze_prim->name();
  auto axis = squeeze_prim->get_axis();
  std::vector<int64_t> infer_shape;

  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShape("input_shape", input_args[0]->GetShapeTrack(), op_name);
  auto len = SizeToLong(in_shape.size());
  if (axis.empty()) {
    std::copy_if(in_shape.begin(), in_shape.end(), std::back_inserter(infer_shape),
                 [](int64_t value) { return value != 1; });
  } else {
    for (auto &item : axis) {
      CheckAndConvertUtils::CheckInRange<int64_t>("axis_or_elememt", item, kIncludeBoth, {-len, len + 1}, op_name);
      auto idx = item >= 0 ? item : len + item;
      if (in_shape[idx] != 1) {
        MS_EXCEPTION(ValueError) << "Cannot select an axis to squeeze out which has size not equal to one.";
      }
    }
    for (int64_t i = 0; i < len; i++) {
      auto it = std::find(axis.begin(), axis.end(), i);
      auto it2 = std::find(axis.begin(), axis.end(), i - len);
      if (!(it != axis.end() || it2 != axis.end())) {
        infer_shape.push_back(in_shape[i]);
      }
    }
  }
  return std::make_shared<abstract::Shape>(infer_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](AbstractBasePtr a) { return a == nullptr; })) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  return input_args[0]->BuildType();
}
}  // namespace
AbstractBasePtr SqueezeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameSqueeze, Squeeze);
}  // namespace ops
}  // namespace mindspore
