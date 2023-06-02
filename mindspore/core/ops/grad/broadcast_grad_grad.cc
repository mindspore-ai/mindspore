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

#include <set>
#include <vector>
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/grad/maximum_grad_grad.h"
#include "ops/grad/minimum_grad_grad.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr BroadcastGradGradInferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr int64_t input_num = 4;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto broadcast = BroadCastInferShape(prim_name, {input_args[kInputIndex0], input_args[kInputIndex1]});
  auto x1 = input_args[kInputIndex0]->BuildShape();
  auto x2 = input_args[kInputIndex1]->BuildShape();
  auto dx1 = input_args[kInputIndex2]->BuildShape();
  auto dx2 = input_args[kInputIndex3]->BuildShape();
  auto x1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x1)[kShape];
  auto x2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x2)[kShape];
  auto dx1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(dx1)[kShape];
  auto dx2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(dx2)[kShape];
  auto broadcast_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(broadcast)[kShape];
  auto is_dy_rank =
    IsDynamicRank(x1_shape) || IsDynamicRank(dx1_shape) || IsDynamicRank(x2_shape) || IsDynamicRank(dx2_shape);
  if ((ObscureShapeEqual(x1_shape, dx1_shape) && ObscureShapeEqual(x2_shape, dx2_shape)) || is_dy_rank) {
    auto obscure_broadcast = std::make_shared<abstract::Shape>(broadcast_shape);
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{x1, x2, obscure_broadcast});
  }

  MS_EXCEPTION(ValueError)
    << "For '" << prim_name
    << "', Its input 'grad_x1', 'grad_x2' should have same shape and equal to x1 and x2 shape, but got 'x1' shape:"
    << x1_shape << " vs 'grad_x1' shape: " << dx1_shape << ", 'x2' shape:" << x2_shape
    << " vs 'grad_x2' shape: " << dx2_shape;
}

TuplePtr BroadcastGradGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr int64_t input_num = 4;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  auto x1 = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
  auto x2 = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex1);
  auto dy1 = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex2);
  auto dy2 = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex3);
  (void)abstract::CheckDtypeSame(prim_name, x1, x2);
  (void)abstract::CheckDtypeSame(prim_name, dy1, dy2);
  (void)abstract::CheckDtypeSame(prim_name, x1, dy1);
  auto x_type = input_args[kInputIndex0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type);
  const std::set<TypePtr> broadcast_grad_grad_valid_types = {kInt32, kInt64, kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, broadcast_grad_grad_valid_types, prim_name);
  std::vector<TypePtr> type_tuple{x_type, x_type, x_type};
  return std::make_shared<Tuple>(type_tuple);
}
}  // namespace

MIND_API_OPERATOR_IMPL(MinimumGradGrad, BaseOperator);
MIND_API_OPERATOR_IMPL(MaximumGradGrad, BaseOperator);
abstract::AbstractBasePtr BroadcastGradGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                 const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infer_type = BroadcastGradGradInferType(primitive, input_args);
  auto infer_shape = BroadcastGradGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
void MaximumGradGrad::Init(const bool grad_x, const bool grad_y) {
  set_grad_x(grad_x);
  set_grad_y(grad_y);
}

void MaximumGradGrad::set_grad_x(const bool grad_x) { (void)this->AddAttr(kGradX, api::MakeValue(grad_x)); }

void MaximumGradGrad::set_grad_y(const bool grad_y) { (void)this->AddAttr(kGradY, api::MakeValue(grad_y)); }

bool MaximumGradGrad::get_grad_x() const {
  auto value_ptr = GetAttr(kGradX);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

bool MaximumGradGrad::get_grad_y() const {
  auto value_ptr = GetAttr(kGradY);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

void MinimumGradGrad::set_grad_x(const bool grad_x) { (void)this->AddAttr(kGradX, api::MakeValue(grad_x)); }

void MinimumGradGrad::set_grad_y(const bool grad_y) { (void)this->AddAttr(kGradY, api::MakeValue(grad_y)); }

bool MinimumGradGrad::get_grad_x() const {
  auto value_ptr = GetAttr(kGradX);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

bool MinimumGradGrad::get_grad_y() const {
  auto value_ptr = GetAttr(kGradY);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

// AG means auto generated
class MIND_API AGBroadcastGradGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return BroadcastGradGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return BroadcastGradGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return BroadcastGradGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MaximumGradGrad, prim::kPrimMaximumGradGrad, AGBroadcastGradGradInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(MinimumGradGrad, prim::kPrimMinimumGradGrad, AGBroadcastGradGradInfer, false);
}  // namespace ops
}  // namespace mindspore
