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
#include "ops/triplet_margin_loss.h"

#include <algorithm>
#include <map>
#include <set>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/base/types.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr TripletMarginLossInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto x = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto positive = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto negative = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto margin = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  if (x.size() >= kDim8 || positive.size() >= kDim8 || negative.size() >= kDim8) {
    MS_EXCEPTION(ValueError) << "For " << op_name
                             << ", dimensions of input x positive and negative must be smaller than 8, x_dim: "
                             << x.size() << ", positive_dim: " << positive.size()
                             << ", negative_dim: " << negative.size() << ".";
  }
  if (x.size() <= kDim1 && positive.size() <= kDim1 && negative.size() <= kDim1) {
    MS_EXCEPTION(ValueError)
      << "For " << op_name
      << ", dimensions of input x, positive and negative cannot be less than 1 at the same time, x_dim: " << x.size()
      << ", positive_dim: " << positive.size() << ", negative_dim: " << negative.size() << ".";
  }
  if (margin.size() != kDim0) {
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
    out_shape.push_back(std::max(std::max(x[i], positive[i]), negative[i]));
    if ((x[i] != out_shape[i] && x[i] != SizeToLong(kDim1)) ||
        (positive[i] != out_shape[i] && positive[i] != SizeToLong(kDim1)) ||
        (negative[i] != out_shape[i] && negative[i] != SizeToLong(kDim1))) {
      MS_EXCEPTION(ValueError) << "For " << op_name << ", inputs' shape can't broadcast.";
    }
  }
  (void)out_shape.erase(out_shape.begin() + 1);
  int64_t reduction;
  CheckAndConvertUtils::GetReductionEnumValue(primitive->GetAttr(kReduction), &reduction);
  mindspore::Reduction reduction_ = static_cast<mindspore::Reduction>(reduction);
  if (reduction_ == REDUCTION_SUM || reduction_ == MEAN) {
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
  (void)types.emplace("x", input_args[kInputIndex0]->BuildType());
  (void)types.emplace("positive", input_args[kInputIndex1]->BuildType());
  (void)types.emplace("negative", input_args[kInputIndex2]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  auto margin = input_args[kInputIndex3]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("margin", margin, valid_types2, op_name);
  auto x_type = input_args[kInputIndex0]->BuildType();
  TypePtr output;
  if (x_type->isa<TensorType>()) {
    auto tensor_type = x_type->cast<TensorTypePtr>();
    x_type = tensor_type->element();
  }
  if (IsIdentidityOrSubclass(x_type, kFloat16)) {
    output = kFloat16;
  } else {
    output = kFloat32;
  }
  return output;
}
}  // namespace

MIND_API_OPERATOR_IMPL(TripletMarginLoss, BaseOperator);
void TripletMarginLoss::set_p(const int64_t p) { (void)this->AddAttr(kP, api::MakeValue(p)); }
void TripletMarginLoss::set_eps(const float eps) { (void)this->AddAttr(kEps, api::MakeValue(eps)); }
void TripletMarginLoss::set_swap(const bool swap) { (void)this->AddAttr(kSwap, api::MakeValue(swap)); }
void TripletMarginLoss::set_reduction(const std::string &reduction) {
  (void)this->AddAttr(kReduction, api::MakeValue(reduction));
}
int64_t TripletMarginLoss::get_p() const { return GetValue<int64_t>(GetAttr(kP)); }
float TripletMarginLoss::get_eps() const { return GetValue<float>(GetAttr(kEps)); }
bool TripletMarginLoss::get_swap() const { return GetValue<bool>(GetAttr(kSwap)); }
std::string TripletMarginLoss::get_reduction() const { return GetValue<std::string>(GetAttr(kReduction)); }
AbstractBasePtr TripletMarginLossInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = TripletMarginLossInferType(primitive, input_args);
  auto infer_shape = TripletMarginLossInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGTripletMarginLossInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return TripletMarginLossInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return TripletMarginLossInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return TripletMarginLossInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(TripletMarginLoss, prim::kPrimTripletMarginLoss, AGTripletMarginLossInfer, false);
}  // namespace ops
}  // namespace mindspore
