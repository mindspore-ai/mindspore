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
#include "ops/sparse_reshape.h"

#include <string>
#include <vector>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr SparseReshapeInferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto indices = input_args[0]->BuildShape();
  auto shape = input_args[1]->BuildShape();
  auto new_shape = input_args[2]->BuildShape();
  auto in0_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices)[kShape];
  auto in1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(shape)[kShape];
  auto in2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(new_shape)[kShape];
  const size_t kone = 1;
  const size_t ktwo = 2;
  if (IsDynamicRank(in0_shape) || IsDynamicRank(in2_shape)) {
    auto unknow_shape_ptr = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{unknow_shape_ptr, unknow_shape_ptr});
  }
  if (in0_shape.size() != ktwo) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ",the shape of input `indices` must be 2-D, but got "
                             << in0_shape.size() << "-D.";
  }

  if (in1_shape.size() != kone) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ",the shape of input `shape` must be 1-D, but got "
                             << in1_shape.size() << "-D.";
  }

  if (in2_shape.size() != kone) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ",the shape of input `new_shape` must be 1-D, but got "
                             << in2_shape.size() << "-D.";
  }

  if (in0_shape[0] != -1 && in0_shape[1] != -1 && in0_shape[1] != in1_shape[0]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the rank of input tensor must match input shape length,"
                             << " but got input tensor rank = " << in0_shape[1]
                             << ", and input shape length = " << in1_shape[0] << ".";
  }
  std::vector<int64_t> out0_shape;
  out0_shape.push_back(in0_shape[0]);
  out0_shape.push_back(in2_shape[0]);

  abstract::ShapePtr y_indices = std::make_shared<abstract::Shape>(out0_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{y_indices, new_shape});
}

TuplePtr SparseReshapeInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", input_args[kInputIndex0]->BuildType(), {kInt64}, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("shape", input_args[kInputIndex1]->BuildType(), {kInt64}, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("new_shape", input_args[kInputIndex2]->BuildType(), {kInt64},
                                                   op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{kInt64, kInt64});
}
}  // namespace

AbstractBasePtr SparseReshapeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto types = SparseReshapeInferType(primitive, input_args);
  auto shapes = SparseReshapeInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

MIND_API_OPERATOR_IMPL(SparseReshape, BaseOperator);

// AG means auto generated
class MIND_API AGSparseReshapeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseReshapeInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseReshapeInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseReshapeInfer(engine, primitive, input_args);
  }
  std::set<int64_t> GetValueDependArgIndices() const override { return {1, 2}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseReshape, prim::kPrimSparseReshape, AGSparseReshapeInfer, false);
}  // namespace ops
}  // namespace mindspore
