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

#include "ops/softmax.h"

#include <memory>
#include <set>
#include <vector>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_name.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSoftmaxMinInputSize = 1;
constexpr auto kNameSoftmaxMaxInputSize = 2;

void Softmax::set_axis(const std::vector<int64_t> &axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }

std::vector<int64_t> Softmax::get_axis() const {
  auto value_ptr = GetAttr(kAxis);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Softmax::Init(const int64_t axis) {
  auto op_name = this->name();
  std::vector<int64_t> axis_vec = {axis};
  (void)CheckAndConvertUtils::CheckInteger("axis_len", SizeToLong(axis_vec.size()), kEqual, 1, op_name);
  auto rank = SizeToLong(axis_vec.size());
  for (auto &item : axis_vec) {
    CheckAndConvertUtils::CheckInRange<int64_t>("axis", item, kIncludeLeft, {-rank, rank}, op_name);
  }
  this->set_axis(axis_vec);
}

namespace {
abstract::ShapePtr SoftMaxInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  if (input_args.size() < kNameSoftmaxMinInputSize || input_args.size() > kNameSoftmaxMaxInputSize) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the input args size should be " << kNameSoftmaxMinInputSize
                      << " or " << kNameSoftmaxMaxInputSize << " , but get " << input_args.size();
  }

  auto op_name = primitive->name();
  auto axis = GetValue<std::vector<int64_t>>(primitive->GetAttr(kAxis));
  (void)CheckAndConvertUtils::CheckValue<size_t>("length of axis", axis.size(), kGreaterEqual, 1, op_name);
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  if (shape_map.empty()) {
    // Scalar input, has no shape
    return std::make_shared<abstract::Shape>(std::vector<int64_t>());
  }
  auto in_shape = shape_map[kShape];
  if (IsDynamicRank(in_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }
  auto rank = SizeToLong(in_shape.size());
  for (auto &item : axis) {
    CheckAndConvertUtils::CheckInRange<int64_t>("axis", item, kIncludeLeft, {-rank, rank}, op_name);
  }
  return std::make_shared<abstract::Shape>(in_shape);
}

TypePtr SoftMaxInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  if (input_args.size() < kNameSoftmaxMinInputSize || input_args.size() > kNameSoftmaxMaxInputSize) {
    MS_LOG(EXCEPTION) << "For '" << prim->name() << "', the input args size should be " << kNameSoftmaxMinInputSize
                      << " or " << kNameSoftmaxMaxInputSize << " , but get " << input_args.size();
  }
  if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr &a) { return a == nullptr; })) {
    MS_LOG(EXCEPTION) << "For '" << prim->name()
                      << ", the input args used for infer shape and type is necessary, but missing it.";
  }
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  return CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, prim->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(Softmax, BaseOperator);
AbstractBasePtr SoftmaxInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(SoftMaxInferShape(primitive, input_args), SoftMaxInferType(primitive, input_args));
}

// AG means auto generated
class MIND_API AGSoftmaxInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SoftMaxInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SoftMaxInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SoftmaxInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Softmax, prim::kPrimSoftmax, AGSoftmaxInfer, false);
}  // namespace ops
}  // namespace mindspore
