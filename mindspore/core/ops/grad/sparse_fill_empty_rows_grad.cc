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

#include "ops/grad/sparse_fill_empty_rows_grad.h"

#include <set>
#include <vector>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
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
abstract::TupleShapePtr SparseFillEmptyRowsGradInferShape(const PrimitivePtr &primitive,
                                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto prim_name = primitive->name();
  constexpr size_t number_one = 1;
  auto map_shape_dtype = input_args[kInputIndex0]->BuildShape();
  auto map_shape_vec = CheckAndConvertUtils::ConvertShapePtrToShapeMap(map_shape_dtype)[kShape];
  auto d_value_dtype = input_args[kInputIndex1]->BuildShape();
  auto grad_values_shape_vec = CheckAndConvertUtils::ConvertShapePtrToShapeMap(d_value_dtype)[kShape];
  if (IsDynamicRank(map_shape_vec) || IsDynamicRank(grad_values_shape_vec)) {
    abstract::ShapePtr map_shape_dyn = std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
    abstract::ShapePtr grad_values_shape_dyn = std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{map_shape_dyn, grad_values_shape_dyn});
  }
  (void)CheckAndConvertUtils::CheckInteger("dim of 'reverse_index_map'", SizeToLong(map_shape_vec.size()), kEqual,
                                           number_one, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("dim of 'grad_values'", SizeToLong(grad_values_shape_vec.size()), kEqual,
                                           number_one, prim_name);

  std::vector<abstract::BaseShapePtr> out_shape;
  ShapeVector d_values = {map_shape_vec[0]};
  ShapeVector d_default_value = {};
  out_shape.push_back(std::make_shared<abstract::Shape>(d_values));
  out_shape.push_back(std::make_shared<abstract::Shape>(d_default_value));
  return std::make_shared<abstract::TupleShape>(out_shape);
}
TypePtr SparseFillEmptyRowsGradInferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  const std::set<TypePtr> common_valid_types_with_bool_and_complex = {
    kInt8,   kInt16,   kInt32,   kInt64,   kUInt8, kUInt16,    kUInt32,
    kUInt64, kFloat16, kFloat32, kFloat64, kBool,  kComplex64, kComplex128};
  auto reverse_index_map_type = input_args[kInputIndex0]->BuildType();
  auto grad_values_type = input_args[kInputIndex1]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("reverse_index_map", reverse_index_map_type, {kInt64}, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("grad_values", grad_values_type,
                                                   common_valid_types_with_bool_and_complex, op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{grad_values_type, grad_values_type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(SparseFillEmptyRowsGrad, BaseOperator);
AbstractBasePtr SparseFillEmptyRowsGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto infer_type = SparseFillEmptyRowsGradInferType(primitive, input_args);
  auto infer_shape = SparseFillEmptyRowsGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGSparseFillEmptyRowsGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseFillEmptyRowsGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseFillEmptyRowsGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseFillEmptyRowsGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseFillEmptyRowsGrad, prim::kPrimSparseFillEmptyRowsGrad,
                                 AGSparseFillEmptyRowsGradInfer, false);
}  // namespace ops
}  // namespace mindspore
