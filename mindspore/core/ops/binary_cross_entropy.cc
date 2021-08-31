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

#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include <map>

#include "ops/binary_cross_entropy.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr BinaryCrossEntroyInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInRange("binary_cross_entropy_infer", input_args.size(), kIncludeBoth, {2, 3}, prim_name);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto y_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto weight_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  CheckAndConvertUtils::Check("x shape", x_shape, kEqual, "y shape", y_shape, prim_name);
  std::vector<int64_t> infer_shape;
  if (weight_shape.size() < 1) {
    CheckAndConvertUtils::Check("x shape", y_shape, kEqual, "weight shape", weight_shape, prim_name);
  }
  auto reduction = Reduction(GetValue<int64_t>(primitive->GetAttr(kReduction)));
  if (reduction != REDUCTION_SUM && reduction != MEAN) {
    infer_shape = {x_shape.begin(), infer_shape.end()};
  }
  return std::make_shared<abstract::Shape>(infer_shape);
}

TypePtr BinaryCrossEntroyInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("binary_cross_entropy_infer", SizeToLong(input_args.size()), kEqual,
                                           input_num, prim->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x_shape", input_args[kInputIndex0]->BuildType());
  (void)types.emplace("y_shape", input_args[kInputIndex1]->BuildType());
  auto infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  if (input_args[kInputIndex3]->BuildType() != nullptr) {
    (void)types.emplace("x_shape", input_args[kInputIndex0]->BuildType());
    (void)types.emplace("weight_shape", input_args[kInputIndex2]->BuildType());
    infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  }
  return infer_type;
}
}  // namespace

void BinaryCrossEntropy::set_reduction(const Reduction &reduction) {
  int64_t swi = reduction;
  (void)this->AddAttr(kReduction, MakeValue(swi));
}

Reduction BinaryCrossEntropy::get_reduction() const {
  auto value_ptr = GetAttr(kReduction);
  return Reduction(GetValue<int64_t>(value_ptr));
}
void BinaryCrossEntropy::Init(const Reduction &reduction) { this->set_reduction(reduction); }

AbstractBasePtr BinaryCrossEntropyInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(BinaryCrossEntroyInferType(primitive, input_args),
                                                    BinaryCrossEntroyInferShape(primitive, input_args));
}
REGISTER_PRIMITIVE_C(kNameBinaryCrossEntropy, BinaryCrossEntropy);
}  // namespace ops
}  // namespace mindspore
