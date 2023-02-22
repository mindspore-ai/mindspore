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

#include "ops/list_diff.h"

#include <memory>
#include <set>
#include <vector>

#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
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
abstract::TupleShapePtr ListDiffInferShape(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) {
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto y_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  if (IsDynamicRank(x_shape) || IsDynamicRank(y_shape)) {
    abstract::ShapePtr rank_shape = std::make_shared<abstract::Shape>(ShapeVector({-2}));
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{rank_shape, rank_shape});
  }
  bool shape_check_pass = x_shape.size() == kDim1 && y_shape.size() == kDim1;
  if (!shape_check_pass) {
    MS_EXCEPTION(ValueError) << "For ListDiff, input x, y should be 1D, but get x dims = " << x_shape.size()
                             << ", y dims = " << y_shape.size() << ".";
  }
  int64_t max_size = x_shape[kInputIndex0];
  ShapeVector out_shape_dynamic = {abstract::Shape::kShapeDimAny};
  ShapeVector out_max_shape = {max_size};
  abstract::ShapePtr out_shape = std::make_shared<abstract::Shape>(out_shape_dynamic, out_max_shape);
  abstract::ShapePtr idx_shape = std::make_shared<abstract::Shape>(out_shape_dynamic, out_max_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape, idx_shape});
}

TuplePtr ListDiffInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kUInt8, kUInt16, kInt8, kInt16, kInt32, kInt64};
  auto x_type =
    CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[kInputIndex0]->BuildType(), valid_types, op_name);
  auto y_type =
    CheckAndConvertUtils::CheckTensorTypeValid("y", input_args[kInputIndex1]->BuildType(), valid_types, op_name);
  if (!(x_type->equal(y_type))) {
    MS_EXCEPTION(TypeError) << "For ListDiff, type of 'x' and 'y' should be same. But get x[" << x_type->ToString()
                            << "], y[" << y_type->ToString() << "].";
  }
  auto idx_type_value_ptr = primitive->GetAttr(kOutIdx);
  MS_EXCEPTION_IF_NULL(idx_type_value_ptr);
  auto idx_type = idx_type_value_ptr->cast<TypePtr>();
  MS_EXCEPTION_IF_NULL(idx_type);
  (void)CheckAndConvertUtils::CheckSubClass(kOutIdx, idx_type, {kInt32, kInt64}, op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_type, idx_type});
}
}  // namespace

AbstractBasePtr ListDiffInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  auto types = ListDiffInferType(primitive, input_args);
  auto shapes = ListDiffInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

MIND_API_OPERATOR_IMPL(ListDiff, BaseOperator);

// AG means auto generated
class MIND_API AGListDiffInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ListDiffInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ListDiffInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ListDiffInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ListDiff, prim::kPrimListDiff, AGListDiffInfer, false);
}  // namespace ops
}  // namespace mindspore
