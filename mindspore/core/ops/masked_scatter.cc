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

#include <map>
#include <string>
#include <set>

#include "ops/masked_scatter.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr MaskedScatterInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto x_shape = x_shape_map[kShape];
  auto mask_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape());
  auto mask_shape = mask_shape_map[kShape];
  CheckAndConvertUtils::CheckInteger("dim of input_x", x_shape.size(), kGreaterEqual, mask_shape.size(), op_name);

  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr MaskedScatterInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = prim->name();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("mask", input_args[1]->BuildType(), {kBool}, op_name);
  std::set<TypePtr> valid_types;
  valid_types = {kFloat16, kFloat32, kFloat64, kUInt8, kInt8, kInt16, kInt32, kInt64};
  auto x_type = input_args[kInputIndex0]->BuildType();
  auto updates_type = input_args[kInputIndex2]->BuildType();
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", x_type);
  (void)types.emplace("updates", updates_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("updates", updates_type, valid_types, op_name);
  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(MaskedScatter, BaseOperator);
AbstractBasePtr MaskedScatterInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kMaskedScaterInputsNum = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kMaskedScaterInputsNum, primitive->name());
  auto infer_type = MaskedScatterInferType(primitive, input_args);
  auto infer_shape = MaskedScatterInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGMaskedScatterInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MaskedScatterInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MaskedScatterInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MaskedScatterInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MaskedScatter, prim::kPrimMaskedScatter, AGMaskedScatterInfer, false);
}  // namespace ops
}  // namespace mindspore
