/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/masked_fill.h"

#include <map>
#include <set>
#include <string>

#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr MaskedFillInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  auto input_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto input_shape = input_shape_map[kShape];
  auto mask_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape());
  auto mask_shape = mask_shape_map[kShape];
  auto value_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape());
  auto value_shape = value_shape_map[kShape];
  auto broadcast_shape = CalBroadCastShape(input_shape, mask_shape, op_name, "input", "mask");
  int64_t batch_rank = 0;
  if (primitive->HasAttr(kBatchRank)) {
    auto value_ptr = primitive->GetAttr(kBatchRank);
    batch_rank = GetValue<int64_t>(value_ptr);
  }
  if (batch_rank == 0 && value_shape.size() != 0) {
    MS_EXCEPTION(ValueError)
      << "For '" << op_name
      << "', 'value' only supports a 0-dimensional value tensor or a float number, but got tensor with "
      << value_shape.size() << " dimension(s).";
  } else if (value_shape.size() != 0) {
    (void)CheckAndConvertUtils::CheckInteger("value shape size", SizeToLong(value_shape.size()), kEqual, batch_rank,
                                             op_name);
    (void)CheckAndConvertUtils::CheckInteger("value shape size", SizeToLong(value_shape.size()), kLessEqual,
                                             SizeToLong(broadcast_shape.size()), op_name);
    for (size_t i = 0; i < LongToSize(batch_rank); i++) {
      if (value_shape[i] != broadcast_shape[i]) {
        MS_EXCEPTION(ValueError) << "For '" << op_name << "', the " << i
                                 << "th index of value shape should be equal to " << broadcast_shape[i] << ", but got "
                                 << value_shape[i];
      }
    }
  }

  return std::make_shared<abstract::Shape>(broadcast_shape);
}

TypePtr MaskedFillInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto op_name = prim->name();
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("mask", input_args[1]->BuildType(), {kBool}, op_name);
  std::set<TypePtr> valid_types;
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool is_ascend = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
  if (is_ascend) {
    valid_types = {kBool, kUInt8, kInt8, kInt16, kInt32, kInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  } else {
    valid_types = {kBool,    kInt8,    kInt16,   kInt32, kInt64, kUInt8, kUInt16,    kUInt32,    kUInt64,
                   kFloat16, kFloat32, kFloat64, kInt,   kUInt,  kFloat, kComplex64, kComplex128};
  }
  if (input_args[kInputIndex2]->isa<abstract::AbstractTensor>()) {
    std::map<std::string, TypePtr> types;
    (void)types.emplace("input", input_args[kInputIndex0]->BuildType());
    (void)types.emplace("value", input_args[kInputIndex2]->BuildType());
    (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
    return types["input"];
  } else {
    (void)CheckAndConvertUtils::CheckSubClass("value", input_args[kInputIndex2]->BuildType(), {kFloat}, op_name);
    auto input_type = input_args[kInputIndex0]->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("input", input_type, valid_types, op_name);
    return input_type;
  }
}
}  // namespace

MIND_API_OPERATOR_IMPL(MaskedFill, BaseOperator);
AbstractBasePtr MaskedFillInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto infer_type = MaskedFillInferType(primitive, input_args);
  auto infer_shape = MaskedFillInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGMaskedFillInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MaskedFillInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MaskedFillInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MaskedFillInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MaskedFill, prim::kPrimMaskedFill, AGMaskedFillInfer, false);
}  // namespace ops
}  // namespace mindspore
