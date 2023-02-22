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

#include <set>
#include <vector>
#include <memory>

#include "ops/sparse_tensor_dense_add.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/primitive_c.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
abstract::ShapePtr SparseTensorDenseAddInferShape(const PrimitivePtr &prim,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  auto x1_indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto x1_values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto x1_shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->BuildShape())[kShape];
  auto x2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[3]->BuildShape())[kShape];
  size_t x1_indices_shape_size = x1_indices_shape.size();
  size_t x1_values_shape_size = x1_values_shape.size();
  size_t x1_shape_shape_size = x1_shape_shape.size();
  size_t x2_shape_size = x2_shape.size();
  const size_t kDimensionOne = 1;
  const size_t kDimensionTwo = 2;
  const size_t kDimensionFive = 5;

  if (x1_values_shape_size == 0 || x2_shape_size == 0) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", the 'x1_values_shape' or 'x2_shape' cannot be scalar ";
  }

  if (!IsDynamicRank(x1_indices_shape) && x1_indices_shape_size != kDimensionTwo) {
    MS_EXCEPTION(ValueError) << "For " << prim_name
                             << ", the 'x1_indices' should have rank 2, but got: " << x1_indices_shape_size;
  }
  if (!IsDynamicRank(x1_shape_shape) && x1_shape_shape_size != kDimensionOne) {
    MS_EXCEPTION(ValueError) << "For " << prim_name
                             << ", the 'x1_shape' should have rank 1, but got: : " << x1_shape_shape_size;
  }
  if (!IsDynamic(x1_values_shape) && !IsDynamic(x1_shape_shape) && !IsDynamic(x1_indices_shape) &&
      !IsDynamic(x2_shape)) {
    if (x1_values_shape_size != kDimensionOne || x1_values_shape[0] != x1_indices_shape[0]) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', the 'x1_values' must be a 1-D tensor and the first dimension length"
                               << " must be equal to the first dimension length of 'x1_indices', but got "
                               << x1_values_shape[0] << " vs " << x1_indices_shape[0] << ".";
    }
    if (x1_shape_shape[0] != x1_indices_shape[1]) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', the length of 'x1_shape' should be equal to the second dimension"
                               << " length of 'x1_indices', but got " << x1_shape_shape[0] << " vs "
                               << x1_indices_shape[1] << ".";
    }
    size_t x1_shape_rank = static_cast<size_t>(x1_shape_shape[0]);
    if (x1_shape_rank != x2_shape_size) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "',  the rank of 'x1_shape' should be equal to the rank of 'x2_shape', but got "
                               << x1_shape_rank << " vs " << x2_shape_size << ".";
    }
    if (x2_shape_size > kDimensionFive || x2_shape_size < kDimensionOne) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "',  Only tensors with ranks between 1 and 5 are currently supported. "
                               << "Tensor rank: " << x2_shape_size << ".";
    }
  }
  ShapeVector output_shape = x2_shape;
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr SparseTensorDenseAddInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto indices_type = input_args[kInputIndex0]->BuildType();
  auto values_type = input_args[kInputIndex1]->BuildType();
  auto shape_type = input_args[kInputIndex2]->BuildType();
  auto x2_type = input_args[kInputIndex3]->BuildType();
  const std::set<TypePtr> valid_indices_types = {kInt32, kInt64};
  const std::set<TypePtr> valid_values_types = {kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,    kUInt32,
                                                kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  (void)CheckAndConvertUtils::CheckTensorTypeSame({{"indices", indices_type}, {"shape", shape_type}},
                                                  valid_indices_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeSame({{"values", values_type}, {"x2", x2_type}}, valid_values_types,
                                                  prim->name());
  return x2_type;
}

AbstractBasePtr SparseTensorDenseAddInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &prim,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  constexpr int inputs_num = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, inputs_num, prim->name());
  auto infer_type = SparseTensorDenseAddInferType(prim, input_args);
  auto infer_shape = SparseTensorDenseAddInferShape(prim, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
MIND_API_OPERATOR_IMPL(SparseTensorDenseAdd, BaseOperator);

// AG means auto generated
class MIND_API AGSparseTensorDenseAddInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseTensorDenseAddInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseTensorDenseAddInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseTensorDenseAddInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseTensorDenseAdd, prim::kPrimSparseTensorDenseAdd, AGSparseTensorDenseAddInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
