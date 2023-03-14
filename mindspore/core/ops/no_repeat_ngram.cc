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

#include "ops/no_repeat_ngram.h"

#include <string>
#include <map>
#include <set>
#include <vector>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kNoRepeatNGramParamValue = 1;
}  // namespace
namespace {
abstract::ShapePtr NoRepeatNGramInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto ngram_size = GetValue<int64_t>(primitive->GetAttr(kNgramSize));
  const int64_t kShapeSize = 3;
  constexpr int64_t kIndex0 = 0;
  constexpr int64_t kIndex1 = 1;
  constexpr int64_t kIndex2 = 2;
  auto state_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto state_shape = state_shape_map[kShape];
  auto log_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
  auto log_shape = log_shape_map[kShape];
  if (IsDynamicRank(log_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  (void)CheckAndConvertUtils::CheckInteger("rank of state_seq", SizeToLong(state_shape.size()), kEqual, kShapeSize,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("rank of log_probs", SizeToLong(log_shape.size()), kEqual, kShapeSize,
                                           prim_name);
  if (IsDynamicShape(state_shape)) {
    std::vector<int64_t> output_shape(state_shape.size(), -1);
    return std::make_shared<abstract::Shape>(output_shape);
  }
  if (IsDynamicShape(log_shape)) {
    return std::make_shared<abstract::Shape>(log_shape);
  }
  (void)CheckAndConvertUtils::CheckValue("state_seq shape[0]", state_shape.at(kIndex0), kEqual, "log_probs shape[0]",
                                         log_shape.at(kIndex0), prim_name);
  (void)CheckAndConvertUtils::CheckValue("state_seq shape[1]", state_shape.at(kIndex1), kEqual, "log_probs shape[1]",
                                         log_shape.at(kIndex1), prim_name);
  (void)CheckAndConvertUtils::CheckValue("ngram_size", ngram_size, kLessEqual, "state_seq shape[2] + 1",
                                         state_shape.at(kIndex2), prim_name);
  if ((ngram_size < kNoRepeatNGramParamValue)) {
    MS_EXCEPTION(ValueError) << "Param ngram_size  << " << ngram_size << " is illegal. ";
  }

  return std::make_shared<abstract::Shape>(log_shape);
}
TypePtr NoRepeatNGramInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::map<std::string, TypePtr> seq_types;
  (void)seq_types.emplace("seq_type", input_args[0]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(seq_types, {kInt32}, prim->name());
  std::set<TypePtr> valid_params_types = {kFloat16, kFloat32, kFloat64};
  std::map<std::string, TypePtr> log_types;
  (void)log_types.emplace("log_types", input_args[1]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(log_types, valid_params_types, prim->name());
}
}  // namespace

void NoRepeatNGram::set_ngram(const int64_t ngram) { (void)this->AddAttr(kNgramSize, api::MakeValue(ngram)); }
/// \brief Get ngram.
///
/// \return ngram.
int64_t NoRepeatNGram::get_ngram() const { return GetValue<int64_t>(GetAttr(kNgramSize)); }

MIND_API_OPERATOR_IMPL(NoRepeatNGram, BaseOperator);
AbstractBasePtr NoRepeatNGramInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto type = NoRepeatNGramInferType(primitive, input_args);
  auto shape = NoRepeatNGramInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

// AG means auto generated
class MIND_API AGNoRepeatNGramInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return NoRepeatNGramInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return NoRepeatNGramInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return NoRepeatNGramInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(NoRepeatNGram, prim::kPrimNoRepeatNGram, AGNoRepeatNGramInfer, false);
}  // namespace ops
}  // namespace mindspore
