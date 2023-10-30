/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/array_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "ops/split_with_size.h"

namespace mindspore {
namespace ops {
namespace {
int64_t CaculateAxis(const AbstractBasePtr &input_abs) {
  auto axis_value = input_abs->BuildValue();
  if (axis_value == nullptr || axis_value->isa<ValueAny>()) {
    MS_LOG(EXCEPTION) << "For SplitWithSize op, axis should be int64_t, but got " << axis_value->ToString();
  }
  auto axis = GetValue<int64_t>(axis_value);
  return axis;
}

std::vector<int64_t> CaculateSplitSize(const AbstractBasePtr &input_abs) {
  auto split_size_value = input_abs->BuildValue()->cast<ValueTuplePtr>();
  if (split_size_value == nullptr || split_size_value->isa<ValueAny>()) {
    MS_LOG(EXCEPTION) << "For SplitWithSize op, split size should be tuple[int], but got "
                      << split_size_value->ToString();
  }
  std::vector<int64_t> split_size;
  std::transform(split_size_value->value().begin(), split_size_value->value().end(), std::back_inserter(split_size),
                 [](const ValuePtr &value) { return GetValue<int64_t>(value); });
  return split_size;
}

abstract::TupleShapePtr SplitWithSizeInferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto shape = input_args[0]->BuildShape();
  auto input_shape_ptr = shape->cast<abstract::ShapePtr>();
  auto input_shape = shape_map[kShape];

  auto axis = CaculateAxis(input_args[kIndex2]);
  size_t pos = LongToSize(axis);

  std::vector<abstract::BaseShapePtr> output_list;
  auto rank = SizeToLong(input_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("rank", rank, kGreaterEqual, 1, prim_name);
  if (axis < 0) {
    axis += rank;
  }
  CheckAndConvertUtils::CheckInRange("axis", axis, kIncludeLeft, {-rank, rank}, prim_name);

  auto split_size = CaculateSplitSize(input_args[kIndex1]);
  int64_t sum_split_size = std::accumulate(split_size.begin(), split_size.end(), 0);
  (void)CheckAndConvertUtils::CheckInteger("sum_split_size", sum_split_size, kEqual, input_shape[pos], prim_name);
  auto output_shape = input_shape;
  for (const int64_t &size : split_size) {
    output_shape[pos] = size;
    abstract::ShapePtr output = std::make_shared<abstract::Shape>(output_shape);
    (void)output_list.emplace_back(output);
  }
  return std::make_shared<abstract::TupleShape>(output_list);
}

TuplePtr SplitWithSizeInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  auto split_size = CaculateSplitSize(input_args[kIndex1]);
  auto infer_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(infer_type);
  static const std::set<TypePtr> valid_types = {kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,     kUInt32,
                                                kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128, kBool};
  auto type = CheckAndConvertUtils::CheckTensorTypeValid("input", infer_type, valid_types, prim->name());
  std::vector<TypePtr> type_tuple;
  for (size_t i = 0; i < split_size.size(); i++) {
    (void)type_tuple.emplace_back(type);
  }
  return std::make_shared<Tuple>(type_tuple);
}
}  // namespace

MIND_API_OPERATOR_IMPL(SplitWithSize, BaseOperator);
AbstractBasePtr SplitWithSizeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  const int64_t kInputNum = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto infertype = SplitWithSizeInferType(primitive, input_args);
  auto infershape = SplitWithSizeInferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}

// AG means auto generated
class MIND_API AGSplitWithSizeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SplitWithSizeInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SplitWithSizeInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SplitWithSizeInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SplitWithSize, prim::kPrimSplitWithSize, AGSplitWithSizeInfer, false);
}  // namespace ops
}  // namespace mindspore
