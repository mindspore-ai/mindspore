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

#include "ops/renorm.h"

#include <map>
#include <set>
#include <string>
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/nn_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kInputSize = 1;
constexpr auto kAttrRecomputeShape = "RecomputeShape";
constexpr auto kAttrDim = "dim";

ShapeVector FreshRenormInferShape(const PrimitivePtr &prim, const ShapeVector &in_shape) {
  MS_EXCEPTION_IF_NULL(prim);
  auto dim = GetValue<int64_t>(prim->GetAttr(kAttrDim));
  if (dim > 0 && dim >= SizeToLong(in_shape.size())) {
    MS_LOG(EXCEPTION) << "Attr dim must be less than the shape size, but got dim:" << dim
                      << ", shape size:" << in_shape.size();
  }
  if (dim < 0) {
    if (std::abs(dim) <= SizeToLong(in_shape.size())) {
      dim += SizeToLong(in_shape.size());
      (void)prim->DelAttr(kAttrDim);
      (void)prim->AddAttr(kAttrDim, MakeValue(dim));
    } else {
      MS_LOG(EXCEPTION) << "Attr dim must be less than the shape size, but got dim:" << dim
                        << ", shape size:" << in_shape.size();
    }
  }
  ShapeVector out = in_shape;
  for (size_t i = 0; i < in_shape.size(); i++) {
    if (static_cast<int64_t>(i) != dim) {
      out[i] = 1;
    }
  }
  return out;
}

TypePtr RenormInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto name = prim->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, kInputSize, name);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto x_dtype = input_args[0]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_dtype, valid_types, name);
  return x_dtype->Clone();
}

abstract::ShapePtr RenormInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, kInputSize,
                                           prim_name);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShape());
  auto shape = input_shape[kShape];
  MS_EXCEPTION_IF_ZERO("Renorm input shape", shape.size());
  auto out_shape = shape;
  if (primitive->HasAttr(kAttrRecomputeShape) && GetValue<bool>(primitive->GetAttr(kAttrRecomputeShape))) {
    out_shape = FreshRenormInferShape(primitive, out_shape);
  }

  return std::make_shared<abstract::Shape>(out_shape);
}
}  // namespace

MIND_API_OPERATOR_IMPL(Renorm, BaseOperator);
AbstractBasePtr RenormInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  return abstract::MakeAbstract(RenormInferShape(primitive, input_args), RenormInferType(primitive, input_args));
}

// AG means auto generated
class MIND_API AGRenormInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return RenormInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return RenormInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return RenormInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Renorm, prim::kPrimRenorm, AGRenormInfer, false);
}  // namespace ops
}  // namespace mindspore
