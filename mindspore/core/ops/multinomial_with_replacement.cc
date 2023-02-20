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

#include "ops/multinomial_with_replacement.h"

#include <set>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
ShapeVector output_sizeList;
}  // namespace

void MultinomialWithReplacement::Init(int64_t numsamples, bool replacement) {
  this->set_numsamples(numsamples);
  this->set_replacement(replacement);
}

void MultinomialWithReplacement::set_numsamples(int64_t numsamples) {
  (void)this->AddAttr("numsamples", api::MakeValue(numsamples));
}

int64_t MultinomialWithReplacement::get_numsamples() const {
  auto numsamples = this->GetAttr("numsamples");
  MS_EXCEPTION_IF_NULL(numsamples);
  return GetValue<int64_t>(numsamples);
}

void MultinomialWithReplacement::set_replacement(bool replacement) {
  (void)this->AddAttr("replacement", api::MakeValue(replacement));
}

bool MultinomialWithReplacement::get_replacement() const {
  auto replacement = this->GetAttr("replacement");
  MS_EXCEPTION_IF_NULL(replacement);
  return GetValue<bool>(replacement);
}

abstract::BaseShapePtr MultinomialWithReplacementInferShape(const PrimitivePtr &primitive,
                                                            const std::vector<AbstractBasePtr> &input_args) {
  const int64_t x_rank_max = 2;
  const int64_t x_rank_min = 1;
  const int64_t dyn_shape = abstract::Shape::kShapeDimAny;
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  std::vector<int64_t> y_shape;
  if (x_shape.size() > x_rank_max || x_shape.size() < x_rank_min) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', 'x' must have a rank of 1 or 2, but got rank "
                             << x_shape.size() << ".";
  }
  auto numsamples_ptr = primitive->GetAttr("numsamples");
  auto numsamples = GetValue<int64_t>(numsamples_ptr);
  auto replacement_ptr = primitive->GetAttr("replacement");
  auto replacement = GetValue<bool>(replacement_ptr);

  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  if (x_shape.size() == x_rank_min) {
    numsamples = (x_shape[0] == dyn_shape) ? dyn_shape : numsamples;
    if (replacement == false && x_shape[0] < numsamples) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the value of numsamples must equal or less than x_shape[-1], but got "
                               << numsamples << ".";
    }
    y_shape.push_back(numsamples);
  } else {
    numsamples = (x_shape[1] == dyn_shape) ? dyn_shape : numsamples;
    if (replacement == false && x_shape[1] < numsamples) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the value of numsamples must equal or less than x_shape[-1], but got "
                               << numsamples << ".";
    }
    y_shape.push_back(x_shape[0]);
    y_shape.push_back(numsamples);
  }
  return std::make_shared<abstract::Shape>(y_shape);
}

TypePtr MultinomialWithReplacementInferType(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto x_dtype = input_args[0]->BuildType();
  auto seed_dtype = input_args[1]->BuildType();
  auto offset_dtype = input_args[2]->BuildType();
  TypePtr y_type = {kInt64};
  const std::set<TypePtr> valid_types_x = {kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> valid_types_seed = {kInt64};
  CheckAndConvertUtils::CheckTensorTypeValid("x_dtype", x_dtype, valid_types_x, op_name);
  CheckAndConvertUtils::CheckTensorTypeValid("seed_dtype", seed_dtype, valid_types_seed, op_name);
  CheckAndConvertUtils::CheckTensorTypeValid("offset_dtype", offset_dtype, valid_types_seed, op_name);
  return y_type;
}

MIND_API_OPERATOR_IMPL(MultinomialWithReplacement, BaseOperator);
AbstractBasePtr MultinomialWithReplacementInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto types = MultinomialWithReplacementInferType(primitive, input_args);
  auto shapes = MultinomialWithReplacementInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

// AG means auto generated
class MIND_API AGMultinomialWithReplacementInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MultinomialWithReplacementInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MultinomialWithReplacementInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MultinomialWithReplacementInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MultinomialWithReplacement, prim::kPrimMultinomialWithReplacement,
                                 AGMultinomialWithReplacementInfer, false);
}  // namespace ops
}  // namespace mindspore
