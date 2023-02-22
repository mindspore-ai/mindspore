/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "ops/ctc_loss_v2.h"

#include <vector>
#include <string>
#include <memory>
#include <set>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
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
int64_t CTCLossV2::get_blank() const { return GetValue<int64_t>(GetAttr(kAttrBlank)); }
std::string CTCLossV2::get_reduction() const { return GetValue<std::string>(GetAttr(kAttrReduction)); }
bool CTCLossV2::get_zero_infinity() const { return GetValue<bool>(GetAttr(kAttrZeroInfinity)); }
namespace {
void CheckInputLengthType(const std::string &arg_name, const AbstractBasePtr &input_arg,
                          const std::set<TypePtr> &valid_type, const std::string &prim_name) {
  if (input_arg->isa<abstract::AbstractTensor>()) {
    (void)CheckAndConvertUtils::CheckTypeValid(arg_name, input_arg->BuildType(), valid_type, prim_name);
  } else if (input_arg->isa<abstract::AbstractTuple>()) {
    auto elements = input_arg->cast<abstract::AbstractTuplePtr>()->elements();
    for (size_t i = 0; i < elements.size(); ++i) {
      (void)CheckAndConvertUtils::CheckSubClass(arg_name, elements[i]->BuildType(), valid_type, prim_name);
    }
  } else {
    MS_EXCEPTION(TypeError) << "For primitive[" << prim_name << "], the input " << input_arg->type_name()
                            << " must be a tuple or a tensor with all Int elements, but got " << input_arg->ToString()
                            << ".";
  }
}
abstract::TupleShapePtr CTCLossV2InferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  constexpr size_t kLenLogProbs = 3;
  constexpr size_t kLenTarget = 2;
  constexpr int64_t kMulti = 2;
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  auto log_probs_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndex0]->BuildShape())[kShape];
  auto targets_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndex1]->BuildShape())[kShape];
  auto input_lengths_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndex2]->BuildShape())[kShape];
  auto target_lengths_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndex3]->BuildShape())[kShape];
  if (input_args[kIndex2]->isa<abstract::AbstractTuple>()) {
    auto size = input_args[kIndex2]->cast<abstract::AbstractTuplePtr>()->elements().size();
    input_lengths_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{SizeToLong(size)})->shape();
  }
  if (input_args[kIndex3]->isa<abstract::AbstractTuple>()) {
    auto size = input_args[kIndex3]->cast<abstract::AbstractTuplePtr>()->elements().size();
    target_lengths_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{SizeToLong(size)})->shape();
  }

  if (IsDynamicRank(log_probs_shape) || IsDynamicRank(targets_shape) || IsDynamicRank(input_lengths_shape) ||
      IsDynamicRank(target_lengths_shape)) {
    std::vector<int64_t> dyn_shape = {abstract::Shape::kShapeRankAny};
    abstract::ShapePtr neg_log_shape = std::make_shared<abstract::Shape>(dyn_shape);
    abstract::ShapePtr log_alpha_shape = std::make_shared<abstract::Shape>(dyn_shape);
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{neg_log_shape, log_alpha_shape});
  }

  (void)CheckAndConvertUtils::CheckValue("dim of log_probs", log_probs_shape.size(), kEqual, kLenLogProbs, prim_name);
  (void)CheckAndConvertUtils::CheckValue("dim of targets", targets_shape.size(), kEqual, kLenTarget, prim_name);

  int64_t T = log_probs_shape[kIndex0];
  int64_t N = log_probs_shape[kIndex1];
  int64_t C = log_probs_shape[kIndex2];
  int64_t S = targets_shape[kIndex1];

  int64_t padded_S = (S == abstract::Shape::kShapeDimAny) ? abstract::Shape::kShapeDimAny : (kMulti * S + 1);
  abstract::ShapePtr neg_log_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{N});
  abstract::ShapePtr log_alpha_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{N, T, padded_S});

  if (IsDynamicShape(log_probs_shape) || IsDynamicShape(targets_shape) || IsDynamicShape(input_lengths_shape) ||
      IsDynamicShape(target_lengths_shape)) {
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{neg_log_shape, log_alpha_shape});
  }

  (void)CheckAndConvertUtils::CheckValue<size_t>("dim of input_lengths", input_lengths_shape.size(), kEqual, kDim1,
                                                 prim_name);
  (void)CheckAndConvertUtils::CheckValue<size_t>("dim of target_lengths", target_lengths_shape.size(), kEqual, kDim1,
                                                 prim_name);
  (void)CheckAndConvertUtils::CheckValue<int64_t>("input_lengths[0]", input_lengths_shape[0], kEqual, N, prim_name);
  (void)CheckAndConvertUtils::CheckValue<int64_t>("target_lengths[0]", target_lengths_shape[0], kEqual, N, prim_name);

  // check blank
  auto blank = GetValue<int64_t>(primitive->GetAttr(kAttrBlank));
  CheckAndConvertUtils::CheckInRange(kAttrBlank, blank, kIncludeLeft, {0, C}, prim_name);

  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{neg_log_shape, log_alpha_shape});
}

TuplePtr CTCLossV2InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto name = primitive->name();
  auto type = CheckAndConvertUtils::CheckTypeValid("log_probs", input_args[kInputIndex0]->BuildType(),
                                                   {kFloat32, kFloat64}, name);
  (void)CheckAndConvertUtils::CheckTypeValid("targets", input_args[kInputIndex1]->BuildType(), {kInt32, kInt64}, name);

  CheckInputLengthType("input_lengths", input_args[kInputIndex2], {kInt32, kInt64}, name);
  CheckInputLengthType("target_lengths", input_args[kInputIndex3], {kInt32, kInt64}, name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{type, type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(CTCLossV2, BaseOperator);
AbstractBasePtr CTCLossV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  constexpr int64_t kInputNum = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto type = CTCLossV2InferType(primitive, input_args);
  auto shape = CTCLossV2InferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

// AG means auto generated
class MIND_API AGCTCLossV2Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return CTCLossV2InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return CTCLossV2InferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return CTCLossV2Infer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(CTCLossV2, prim::kPrimCTCLossV2, AGCTCLossV2Infer, false);
}  // namespace ops
}  // namespace mindspore
