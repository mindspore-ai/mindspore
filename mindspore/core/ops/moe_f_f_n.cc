/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "ops/moe_f_f_n.h"
#include <string>
#include <map>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/other_ops.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kMinInputNumber = 3;
constexpr int64_t kMaxInputNumber = 10;
constexpr int64_t kXShapeRank = 2;
constexpr int64_t kW1ShapeRank = 3;
constexpr int64_t kBiasShapeRank = 2;
constexpr int64_t kW2ShapeRank = 3;

void CheckInputsNum(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual,
                                           kMinInputNumber, primitive->name());
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kLessEqual, kMaxInputNumber,
                                           primitive->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(MoeFFN, BaseOperator);
class MoeFFNInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    CheckInputsNum(primitive, input_args);
    auto x_shape = input_args[kIndex0]->BuildShape();
    auto real_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape)[kShape];
    (void)CheckAndConvertUtils::CheckInteger("x shape rank", SizeToLong(real_shape.size()), kEqual, kXShapeRank,
                                             primitive->name());
    auto w1_shape = input_args[kIndex2]->BuildShape();
    auto real_w1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(w1_shape)[kShape];
    (void)CheckAndConvertUtils::CheckInteger("w1 shape rank", SizeToLong(real_w1_shape.size()), kEqual, kW1ShapeRank,
                                             primitive->name());

    auto bias1_shape = input_args[kIndex3]->BuildShape();
    auto real_bias1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(bias1_shape)[kShape];
    (void)CheckAndConvertUtils::CheckInteger("bais1 shape rank", SizeToLong(real_bias1_shape.size()), kEqual,
                                             kBiasShapeRank, primitive->name());

    auto w2_shape = input_args[kIndex4]->BuildShape();
    auto real_w2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(w2_shape)[kShape];
    (void)CheckAndConvertUtils::CheckInteger("w2 shape rank", SizeToLong(real_w2_shape.size()), kEqual, kW2ShapeRank,
                                             primitive->name());
    ShapeVector out_shape = {real_shape[kIndex0], real_shape[kIndex1]};
    return std::make_shared<abstract::Shape>(out_shape);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    CheckInputsNum(primitive, input_args);
    MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
    auto x_type = input_args[kIndex0]->BuildType();
    const std::set<TypePtr> valid_types = {kFloat16, kInt8};
    (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, primitive->name());
    return kFloat16;
  }
};
abstract::AbstractBasePtr MoeFFNInferFunc(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  MoeFFNInfer moe_f_f_n_infer;
  auto type = moe_f_f_n_infer.InferType(primitive, input_args);
  auto shape = moe_f_f_n_infer.InferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_OP_INFER_IMPL(MoeFFN, prim::kPrimMoeFFN, MoeFFNInfer, false);
}  // namespace ops
}  // namespace mindspore
