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
#include "ops/sort.h"

#include <memory>
#include <set>
#include <vector>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
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
void Sort::Init(int64_t axis, bool descending) {
  this->set_axis(axis);
  this->set_descending(descending);
}

void Sort::set_axis(int64_t axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }

int64_t Sort::get_axis() const {
  auto axis = this->GetAttr(kAxis);
  MS_EXCEPTION_IF_NULL(axis);
  return GetValue<int64_t>(axis);
}

void Sort::set_descending(bool descending) { (void)this->AddAttr(kDescending, api::MakeValue(descending)); }

bool Sort::get_descending() const {
  auto descending = this->GetAttr(kDescending);
  MS_EXCEPTION_IF_NULL(descending);
  return GetValue<bool>(descending);
}

namespace {
abstract::TupleShapePtr SortInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (IsDynamicRank(x_shape)) {
    auto unknown_shape_ptr = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{unknown_shape_ptr, unknown_shape_ptr});
  }
  if (IsDynamicShape(x_shape)) {
    auto unknown_shape_ptr = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeDimAny});
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{unknown_shape_ptr, unknown_shape_ptr});
  }
  auto x_rank = SizeToLong(x_shape.size());
  auto axis = GetValue<int64_t>(primitive->GetAttr("axis"));
  CheckAndConvertUtils::CheckInRange<int64_t>("axis", axis, kIncludeBoth, {-x_rank, x_rank - 1}, prim_name);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  auto x = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto shape_element = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{shape_element, shape_element});
}

TuplePtr SortInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infer_type = input_args[0]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kUInt8, kInt8, kInt16, kInt32, kInt64, kBool};
  auto type = CheckAndConvertUtils::CheckTensorTypeValid("inputx", infer_type, valid_types, primitive->name());
  std::vector<TypePtr> type_tuple;
  type_tuple.push_back(type);
  type_tuple.push_back(kInt32);
  return std::make_shared<Tuple>(type_tuple);
}
}  // namespace

AbstractBasePtr SortInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kSortInputsNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kSortInputsNum, primitive->name());
  auto infer_type = SortInferType(primitive, input_args);
  auto infer_shape = SortInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(Sort, BaseOperator);

// AG means auto generated
class MIND_API AGSortInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SortInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SortInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SortInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Sort, prim::kPrimSort, AGSortInfer, false);
}  // namespace ops
}  // namespace mindspore
