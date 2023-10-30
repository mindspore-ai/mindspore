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
#include "ops/split_tensor.h"

namespace mindspore {
namespace ops {
namespace {
int64_t CaculateAxis(const AbstractBasePtr &input_abs) {
  auto axis_value = input_abs->BuildValue();
  if (axis_value == nullptr || axis_value->isa<ValueAny>()) {
    MS_LOG(EXCEPTION) << "For SplitTensor op, axis should be int64_t, but got " << axis_value->ToString();
  }
  auto axis = GetValue<int64_t>(axis_value);
  return axis;
}

int64_t CaculateSplitSections(const AbstractBasePtr &input_abs) {
  auto split_size_value = input_abs->BuildValue();
  if (split_size_value == nullptr || split_size_value->isa<ValueAny>()) {
    MS_LOG(EXCEPTION) << "For SplitTensor op, split sections should be int64_t, but got "
                      << split_size_value->ToString();
  }
  auto split_size = GetValue<int64_t>(split_size_value);
  return split_size;
}

abstract::TupleShapePtr SplitTensorInferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto shape = input_args[0]->BuildShape();
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

  auto split_sections = CaculateSplitSections(input_args[kIndex1]);
  auto output_shape = input_shape;
  output_shape[pos] = split_sections;
  for (int64_t i = 0; i < input_shape[pos] / split_sections; ++i) {
    abstract::ShapePtr output = std::make_shared<abstract::Shape>(output_shape);
    (void)output_list.emplace_back(output);
  }
  int64_t last_size = input_shape[pos] % split_sections;
  if (last_size != 0) {
    auto last_shape = input_shape;
    last_shape[pos] = last_size;
    abstract::ShapePtr last_output = std::make_shared<abstract::Shape>(last_shape);
    (void)output_list.push_back(last_output);
  }
  return std::make_shared<abstract::TupleShape>(output_list);
}

TuplePtr SplitTensorInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto shape = input_args[0]->BuildShape();
  auto input_shape = shape_map[kShape];
  auto axis = CaculateAxis(input_args[kIndex2]);
  size_t pos = LongToSize(axis);

  auto split_sections = CaculateSplitSections(input_args[kIndex1]);
  auto output_num = (input_shape[pos] % split_sections) == 0 ? (input_shape[pos] / split_sections)
                                                             : (input_shape[pos] / split_sections) + 1;
  auto infer_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(infer_type);
  static const std::set<TypePtr> valid_types = {kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,     kUInt32,
                                                kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128, kBool};
  auto type = CheckAndConvertUtils::CheckTensorTypeValid("x", infer_type, valid_types, prim->name());
  std::vector<TypePtr> type_tuple;
  for (int32_t i = 0; i < output_num; i++) {
    (void)type_tuple.emplace_back(type);
  }
  return std::make_shared<Tuple>(type_tuple);
}
}  // namespace

MIND_API_OPERATOR_IMPL(SplitTensor, BaseOperator);
AbstractBasePtr SplitTensorInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  const int64_t kInputNum = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto infertype = SplitTensorInferType(primitive, input_args);
  auto infershape = SplitTensorInferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}

// AG means auto generated
class MIND_API AGSplitTensorInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SplitTensorInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SplitTensorInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SplitTensorInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SplitTensor, prim::kPrimSplitTensor, AGSplitTensorInfer, false);
}  // namespace ops
}  // namespace mindspore
