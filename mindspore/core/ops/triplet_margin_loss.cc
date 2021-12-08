/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "ops/triplet_margin_loss.h"
#include <algorithm>
#include "ops/op_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kInputSize = 4;
abstract::ShapePtr TripletMarginLossInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto x = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto positive = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto negative = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto margin = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  const int64_t keight = 8;
  if (x.size() >= keight || positive.size() >= keight || negative.size() >= keight) {
    MS_EXCEPTION(ValueError) << "For " << op_name
                             << ", dimensions of input x positive and negative must be smaller than 8, x_dim: "
                             << x.size() << ", positive_dim: " << positive.size()
                             << ", negative_dim: " << negative.size() << ".";
  }
  const int64_t kone = 1;
  if (x.size() <= kone && positive.size() <= kone && negative.size() <= kone) {
    MS_EXCEPTION(ValueError)
      << "For " << op_name
      << ", dimensions of input x, positive and negative cannot be less than 1 at the same time, x_dim: " << x.size()
      << ", positive_dim: " << positive.size() << ", negative_dim: " << negative.size() << ".";
  }
  if (margin.size() != 0) {
    MS_EXCEPTION(ValueError) << "For " << op_name
                             << ", the dimension of input margin must be 0, margin_dim: " << margin.size() << ".";
  }
  auto dims = std::max(std::max(x.size(), positive.size()), negative.size());
  std::reverse(x.begin(), x.end());
  std::reverse(positive.begin(), positive.end());
  std::reverse(negative.begin(), negative.end());
  x.resize(dims, 1);
  positive.resize(dims, 1);
  negative.resize(dims, 1);
  std::reverse(x.begin(), x.end());
  std::reverse(positive.begin(), positive.end());
  std::reverse(negative.begin(), negative.end());
  ShapeVector out_shape;
  for (size_t i = 0; i < dims; i++) {
    out_shape.push_back((int64_t)std::max(std::max(x[i], positive[i]), negative[i]));
    if ((x[i] != out_shape[i] && x[i] != kone) || (positive[i] != out_shape[i] && positive[i] != kone) ||
        (negative[i] != out_shape[i] && negative[i] != kone)) {
      MS_EXCEPTION(ValueError) << "For " << op_name << ", inputs' shape can't broadcast.";
    }
  }
  out_shape.erase(out_shape.begin() + 1);
  int64_t reduction;
  (void)CheckAndConvertUtils::GetReductionEnumValue(primitive->GetAttr(kReduction), &reduction);
  if (reduction == REDUCTION_SUM || reduction == MEAN) {
    out_shape.resize(0);
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr TripletMarginLossInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  const std::set<TypePtr> valid_types = {kComplex64, kComplex128, kFloat64, kFloat32, kFloat16, kInt16, kInt32,
                                         kInt64,     kInt8,       kUInt16,  kUInt32,  kUInt64,  kUInt8};
  const std::set<TypePtr> valid_types2 = {kFloat32};
  std::map<std::string, TypePtr> types;
  types.emplace("x", input_args[kInputIndex0]->BuildType());
  types.emplace("positive", input_args[kInputIndex1]->BuildType());
  types.emplace("negative", input_args[kInputIndex2]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  auto margin = input_args[kInputIndex3]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("margin", margin, valid_types2, op_name);
  auto x_type = input_args[kInputIndex0]->BuildType();
  TypePtr output;
  if (x_type == kFloat16) {
    output = kFloat16;
  } else {
    output = kFloat32;
  }
  return output;
}
}  // namespace

AbstractBasePtr TripletMarginLossInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 4;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = TripletMarginLossInferType(primitive, input_args);
  auto infer_shape = TripletMarginLossInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(TripletMarginLoss, prim::kPrimTripletMarginLoss, TripletMarginLossInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
