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

#include <set>
#include <string>
#include <vector>
#include <memory>
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/ops/primitive_infer_map.h"

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

  if (in0_shape.size() != ktwo) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ",the shape of input `indices` must be 2-D, but got "
                             << in0_shape.size() << "-D.";
  }

  if (in1_shape.size() != kone) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ",the shape of input `shape` must be 1-D, but got "
                             << in0_shape.size() << "-D.";
  }

  if (in2_shape.size() != kone) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ",the shape of input `new_shape` must be 1-D, but got "
                             << in0_shape.size() << "-D.";
  }

  if (in0_shape[1] != in1_shape[0]) {
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
REGISTER_PRIMITIVE_EVAL_IMPL(SparseReshape, prim::kPrimSparseReshape, SparseReshapeInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
