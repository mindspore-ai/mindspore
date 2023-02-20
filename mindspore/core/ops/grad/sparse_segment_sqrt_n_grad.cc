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

#include "ops/grad/sparse_segment_sqrt_n_grad.h"

#include <map>
#include <set>

#include "abstract/dshape.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/named.h"
#include "ir/primitive.h"
#include "ir/value.h"
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
abstract::ShapePtr SparseSegmentSqrtNGradInferShape(const PrimitivePtr &prim,
                                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto segment_ids_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto output_dim0_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("indices_shape", SizeToLong(indices_shape.size()), kEqual, kInputIndex1,
                                           prim->name());
  (void)CheckAndConvertUtils::CheckInteger("segment_ids_shape", SizeToLong(segment_ids_shape.size()), kEqual,
                                           kInputIndex1, prim->name());
  if (x_shape.size() < kInputIndex1) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', "
                             << "tensor x's rank must be greater than 1, but got [" << x_shape.size() << "].";
  }
  if (!IsDynamic(output_dim0_shape) && output_dim0_shape.size() != kInputIndex0) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', tensor output_dim0 should be a scalar, "
                             << "but got [" << output_dim0_shape.size() << "].";
  }
  if (!(IsDynamic(indices_shape) || IsDynamic(segment_ids_shape)) &&
      indices_shape[kInputIndex0] != segment_ids_shape[kInputIndex0]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the rank of indices and segment_ids must be the same, "
                             << "but got indices [" << indices_shape[kInputIndex0] << "] "
                             << "and segment_ids [" << segment_ids_shape[kInputIndex0] << "].";
  }
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  if (!input_args[kInputIndex3]->BuildValue()->isa<AnyValue>() &&
      !input_args[kInputIndex3]->BuildValue()->isa<None>()) {
    auto output_dim0_value = input_args[kInputIndex3]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(output_dim0_value);
    auto output_dim0_value_ptr = output_dim0_value->BuildValue();
    MS_EXCEPTION_IF_NULL(output_dim0_value_ptr);
    auto output_dim0_value_ptr_tensor =
      CheckAndConvertUtils::CheckTensorIntValue("output_dim0", output_dim0_value_ptr, prim_name);
    auto dim_zero = output_dim0_value_ptr_tensor[kInputIndex0];
    if (dim_zero <= 0) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "' , tensor output_dim0 must > 0, "
                               << "but got [" << dim_zero << "].";
    } else {
      ShapeVector y_shape = x_shape;
      y_shape[kInputIndex0] = static_cast<int64_t>(dim_zero);
      return std::make_shared<abstract::Shape>(y_shape);
    }
  } else {
    ShapeVector output_shape = x_shape;
    output_shape[kInputIndex0] = -1;
    return std::make_shared<abstract::Shape>(output_shape);
  }
}

TypePtr SparseSegmentSqrtNGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto x_type = input_args[kInputIndex0]->BuildType();
  auto indices_type = input_args[kInputIndex1]->BuildType();
  auto segment_ids_type = input_args[kInputIndex2]->BuildType();
  auto output_dim0_type = input_args[kInputIndex3]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> common_valid_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim->name());
  std::map<std::string, TypePtr> types;
  (void)types.emplace("indices", indices_type);
  (void)types.emplace("segment_ids", segment_ids_type);
  (void)types.emplace("output_dim0", output_dim0_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types, prim->name());
  return input_args[kInputIndex0]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(SparseSegmentSqrtNGrad, BaseOperator);
AbstractBasePtr SparseSegmentSqrtNGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &prim,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const int64_t input_num = static_cast<int64_t>(kInputIndex4);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto types = SparseSegmentSqrtNGradInferType(prim, input_args);
  auto shapes = SparseSegmentSqrtNGradInferShape(prim, input_args);
  return abstract::MakeAbstract(shapes, types);
}

// AG means auto generated
class MIND_API AGSparseSegmentSqrtNGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseSegmentSqrtNGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseSegmentSqrtNGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseSegmentSqrtNGradInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {3}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseSegmentSqrtNGrad, prim::kPrimSparseSegmentSqrtNGrad,
                                 AGSparseSegmentSqrtNGradInfer, false);
}  // namespace ops
}  // namespace mindspore
