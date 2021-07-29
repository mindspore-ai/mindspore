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

#include <set>
#include "ops/pad.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto paddings_attr = GetValue<std::vector<std::vector<int64_t>>>(primitive->GetAttr(kPaddings));
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("paddings_size", SizeToLong(paddings_attr.size()), kEqual,
                                           int64_t(2 * x_shape.size()), prim_name);
  int64_t size = SizeToLong(paddings_attr.size());
  for (int64_t i = 0; i < size; i++) {
    for (int64_t j = 0; j < 2; j++) {
      if (paddings_attr[LongToSize(i)][LongToSize(j)] < 0) {
        MS_LOG_ERROR << "All elements of paddings must be >= 0.";
      }
    }
  }
  std::vector<int64_t> out_shape;
  for (int64_t i = 0; i < int64_t(paddings_attr.size() / 2); i++) {
    (void)out_shape.emplace_back(x_shape[LongToSize(i)] + paddings_attr[LongToSize(i)][0] +
                                 paddings_attr[LongToSize(i)][1]);
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> valid_types = {kTensorType};
  return CheckAndConvertUtils::CheckSubClass("infer type", input_args[0]->BuildType(), valid_types, prim->name());
}
}  // namespace

void Pad::Init(const std::vector<std::vector<int64_t>> &paddings) { this->set_paddings(paddings); }
void Pad::set_paddings(const std::vector<std::vector<int64_t>> &paddings) {
  (void)this->AddAttr(kPaddings, MakeValue(paddings));
}
std::vector<std::vector<int64_t>> Pad::get_paddings() const {
  return GetValue<std::vector<std::vector<int64_t>>>(GetAttr(kPaddings));
}
AbstractBasePtr PadInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNamePad, Pad);
}  // namespace ops
}  // namespace mindspore
