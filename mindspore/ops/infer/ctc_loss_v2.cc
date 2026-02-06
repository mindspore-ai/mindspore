/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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

#include "infer/ctc_loss_v2.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ops_utils/op_constants.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "mindapi/ir/value.h"
#include "mindapi/helper.h"
#include "primitive/nn_ops.h"
#include "primitive/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "utils/value_utils.h"
#include "utils/ms_context.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"

namespace mindspore {
namespace ops {
int64_t CTCLossV2::get_blank() const { return GetValue<int64_t>(GetAttr(kAttrBlank)); }
std::string CTCLossV2::get_reduction() const { return GetValue<std::string>(GetAttr(kAttrReduction)); }
bool CTCLossV2::get_zero_infinity() const { return GetValue<bool>(GetAttr(kAttrZeroInfinity)); }
constexpr int64_t kAlignSize = 8;
namespace {
void CheckInputLengthType(const std::string &arg_name, const AbstractBasePtr &input_arg,
                          const std::set<TypePtr> &valid_type, const std::string &prim_name) {
  if (CheckAndConvertUtils::IsTensor(input_arg)) {
    (void)CheckAndConvertUtils::CheckTypeValid(arg_name, input_arg->GetType(), valid_type, prim_name);
  } else if (CheckAndConvertUtils::IsTuple(input_arg)) {
    auto idx_type_ptr = input_arg->GetType();
    MS_EXCEPTION_IF_NULL(idx_type_ptr);
    auto types_list_ptr = idx_type_ptr->cast<TuplePtr>();
    MS_EXCEPTION_IF_NULL(types_list_ptr);
    TypePtrList types_list = types_list_ptr->elements();
    for (size_t i = 0; i < types_list.size(); ++i) {
      (void)CheckAndConvertUtils::CheckSubClass(arg_name, types_list[i], valid_type, prim_name);
    }
  } else {
    MS_EXCEPTION(TypeError) << "For primitive[" << prim_name << "], the input " << input_arg->type_name()
                            << " must be a tuple or a tensor with all Int elements, but got " << input_arg->ToString()
                            << ".";
  }
}
void CalcMaxTargetLengthsValueImpl(int64_t *max_target_lengths_value, const AbstractBasePtr &target_lengths,
                                   const TypeId &dtype) {
  if (dtype == kNumberTypeInt32) {
    const auto &target_lengths_value_opt = GetArrayValue<int>(target_lengths);
    if (!target_lengths_value_opt.has_value()) {
      return;
    }
    const auto &target_lengths_value = target_lengths_value_opt.value();
    for (size_t i = 0; i < target_lengths_value.size(); ++i) {
      if (target_lengths_value[i] > *max_target_lengths_value) {
        *max_target_lengths_value = static_cast<int64_t>(target_lengths_value[i]);
      }
    }
  } else if (dtype == kNumberTypeInt64) {
    const auto &target_lengths_value_opt = GetArrayValue<int64_t>(target_lengths);
    if (!target_lengths_value_opt.has_value()) {
      return;
    }
    const auto &target_lengths_value = target_lengths_value_opt.value();
    for (size_t i = 0; i < target_lengths_value.size(); ++i) {
      if (target_lengths_value[i] > *max_target_lengths_value) {
        *max_target_lengths_value = target_lengths_value[i];
      }
    }
  } else {
    MS_EXCEPTION(TypeError) << "Unsupported data type " << TypeIdToString(dtype) << " for target lengths";
  }
}

void CalcMaxTargetLengthsValue(int64_t *max_target_lengths_value, const AbstractBasePtr &target_lengths) {
  MS_EXCEPTION_IF_NULL(target_lengths);
  const auto &type = target_lengths->GetType();
  MS_EXCEPTION_IF_NULL(type);
  if (type->object_type() == kObjectTypeTuple) {
    auto tuple_type = type->cast<TuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple_type);
    CalcMaxTargetLengthsValueImpl(max_target_lengths_value, target_lengths, tuple_type->elements()[kIndex0]->type_id());
  } else if (type->object_type() == kObjectTypeTensorType) {
    auto tensor_type = type->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type);
    CalcMaxTargetLengthsValueImpl(max_target_lengths_value, target_lengths, tensor_type->element()->type_id());
  } else {
    MS_EXCEPTION(TypeError) << "Target lengths must be a Tuple or Tensor, but got " << type->ToString();
  }
}
abstract::TupleShapePtr CTCLossV2InferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  constexpr size_t kLenLogProbs = 3;
  constexpr size_t kLenTarget = 2;
  constexpr int64_t kMulti = 2;
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  auto log_probs_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndex0]->GetShape())[kShape];
  auto targets_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndex1]->GetShape())[kShape];
  auto input_lengths_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndex2]->GetShape())[kShape];
  auto target_lengths_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndex3]->GetShape())[kShape];
  if (CheckAndConvertUtils::IsTuple(input_args[kIndex2])) {
    auto idx_shape_ptr = input_args[kIndex2]->GetShape();
    MS_EXCEPTION_IF_NULL(idx_shape_ptr);
    auto shape_tuple = idx_shape_ptr->cast<abstract::TupleShapePtr>();
    auto size = shape_tuple->size();
    input_lengths_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{SizeToLong(size)})->shape();
  }
  if (CheckAndConvertUtils::IsTuple(input_args[kIndex3])) {
    auto idx_shape_ptr = input_args[kIndex3]->GetShape();
    MS_EXCEPTION_IF_NULL(idx_shape_ptr);
    auto shape_tuple = idx_shape_ptr->cast<abstract::TupleShapePtr>();
    auto size = shape_tuple->size();
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
  (void)CheckAndConvertUtils::CheckValue<size_t>("dim of input_lengths", input_lengths_shape.size(), kEqual, kDim1,
                                                 prim_name);
  (void)CheckAndConvertUtils::CheckValue<size_t>("dim of target_lengths", target_lengths_shape.size(), kEqual, kDim1,
                                                 prim_name);

  int64_t T = log_probs_shape[kIndex0];
  int64_t N = log_probs_shape[kIndex1];
  int64_t C = log_probs_shape[kIndex2];
  int64_t S = targets_shape[kIndex1];

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  int64_t padded_S = 0;
  if (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    int64_t max_target_lengths_value = INT64_MIN;
    CalcMaxTargetLengthsValue(&max_target_lengths_value, input_args[kIndex3]);
    padded_S = (S == abstract::Shape::kShapeDimAny || max_target_lengths_value == INT64_MIN)
                 ? abstract::Shape::kShapeDimAny
                 : (kMulti * max_target_lengths_value + 1);
    if (padded_S > 0) {
      padded_S = (padded_S + kAlignSize - 1) / kAlignSize * kAlignSize;
    }
  } else {
    padded_S = (S == abstract::Shape::kShapeDimAny) ? abstract::Shape::kShapeDimAny : (kMulti * S + 1);
  }

  abstract::ShapePtr neg_log_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{N});
  abstract::ShapePtr log_alpha_shape = std::make_shared<abstract::Shape>(
    std::vector<int64_t>{N, T, padded_S <= 0 ? abstract::Shape::kShapeDimAny : padded_S});

  if (IsDynamicShape(log_probs_shape) || IsDynamicShape(targets_shape) || IsDynamicShape(input_lengths_shape) ||
      IsDynamicShape(target_lengths_shape)) {
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{neg_log_shape, log_alpha_shape});
  }

  (void)CheckAndConvertUtils::CheckValue<int64_t>("input_lengths.shape[0]", input_lengths_shape[0], kEqual, N,
                                                  prim_name);
  (void)CheckAndConvertUtils::CheckValue<int64_t>("target_lengths.shape[0]", target_lengths_shape[0], kEqual, N,
                                                  prim_name);

  // check blank
  auto blank = GetValue<int64_t>(primitive->GetAttr(kAttrBlank));
  CheckAndConvertUtils::CheckInRange(kAttrBlank, blank, kIncludeLeft, {0, C}, prim_name);

  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{neg_log_shape, log_alpha_shape});
}

TuplePtr CTCLossV2InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto name = primitive->name();
  auto type =
    CheckAndConvertUtils::CheckTypeValid("log_probs", input_args[kInputIndex0]->GetType(), {kFloat32, kFloat64}, name);
  (void)CheckAndConvertUtils::CheckTypeValid("targets", input_args[kInputIndex1]->GetType(), {kInt32, kInt64}, name);

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
class OPS_API AGCTCLossV2Infer : public abstract::OpInferBase {
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
