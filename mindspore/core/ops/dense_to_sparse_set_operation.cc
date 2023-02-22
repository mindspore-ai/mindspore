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

#include "ops/dense_to_sparse_set_operation.h"

#include <set>
#include <vector>
#include <algorithm>
#include <functional>
#include <map>
#include <string>
#include <numeric>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr DenseToSparseSetOperationInferShape(const PrimitivePtr &primitive,
                                                            const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(primitive);
  auto x1_shape_ptr = input_args[0]->BuildShape();
  auto x2_indices_shape_ptr = input_args[1]->BuildShape();
  auto x2_values_shape_ptr = input_args[2]->BuildShape();
  auto x2_shape_shape_ptr = input_args[3]->BuildShape();
  auto x1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x1_shape_ptr)[kShape];
  auto x2_indice_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x2_indices_shape_ptr)[kShape];
  auto x2_values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x2_values_shape_ptr)[kShape];
  auto x2_shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x2_shape_shape_ptr)[kShape];

  // Args x2_indice must be 2D tensor, x2_values and x2_shape must be 1D tensor
  const int64_t tensor2d_num = 2;
  const int64_t tensor1d_num = 1;
  (void)CheckAndConvertUtils::CheckInteger("dimension of 'x2_indices'", x2_indice_shape.size(), kEqual, tensor2d_num,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("dimension of 'x2_values'", x2_values_shape.size(), kEqual, tensor1d_num,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("dimension of 'x2_shape'", x2_shape_shape.size(), kEqual, tensor1d_num,
                                           prim_name);

  // Dimension of x1 must be equal or greater than 2
  (void)CheckAndConvertUtils::CheckInteger("dimension of 'x1'", SizeToLong(x1_shape.size()), kGreaterEqual,
                                           tensor2d_num, prim_name);
  // x2_value shape must be equal to the first dimension of x2_indices
  CheckAndConvertUtils::Check("'x2_values' shape", x2_values_shape[0], kEqual, x2_indice_shape[0], prim_name);

  std::string set_operation_str = GetValue<std::string>(primitive->GetAttr("set_operation"));
  std::transform(set_operation_str.begin(), set_operation_str.end(), set_operation_str.begin(), ::tolower);
  int64_t x1_size = std::accumulate(x1_shape.begin(), x1_shape.end(), 1, std::multiplies<int64_t>());
  int64_t x2_size = SizeToLong(x2_values_shape[0]);

  int64_t y_size_max = 0;

  if (set_operation_str == "a-b") {
    y_size_max = x1_size;
  } else if (set_operation_str == "b-a") {
    y_size_max = x2_size;
  } else if (set_operation_str == "intersection") {
    y_size_max = std::max(x1_size, x2_size);
  } else if (set_operation_str == "union") {
    y_size_max = x1_size + x2_size;
  } else {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the attr 'set_operation' must be one of 'a-b', 'b-a', "
                             << "'intersection', 'union', but get" << set_operation_str << ".";
  }

  // y_indices shape infer
  ShapeVector y_indices_shape = {-1, SizeToLong(x1_shape.size())};
  ShapeVector y_indices_max_shape = {y_size_max, SizeToLong(x1_shape.size())};
  auto y_indices_shape_ptr = std::make_shared<abstract::Shape>(y_indices_shape, y_indices_max_shape);

  // y_values shape infer
  ShapeVector y_values_shape = {-1};
  ShapeVector y_values_max_shape = {y_size_max};
  auto y_values_shape_ptr = std::make_shared<abstract::Shape>(y_values_shape, y_values_max_shape);

  // y_shape shape infer
  ShapeVector y_shape_shape = {SizeToLong(x1_shape.size())};
  auto y_shape_shape_ptr = std::make_shared<abstract::Shape>(y_shape_shape);

  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{y_indices_shape_ptr, y_values_shape_ptr, y_shape_shape_ptr});
}

TuplePtr DenseToSparseSetOperationInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  auto x1_type = input_args[0]->BuildType();
  auto x2_indices_type = input_args[1]->BuildType();
  auto x2_values_type = input_args[2]->BuildType();
  auto x2_shape_type = input_args[3]->BuildType();

  const std::set<TypePtr> valid_types = {kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kString};

  // Args x2_values must have the same type as x1
  std::map<std::string, TypePtr> args;
  (void)args.insert({"x1", x1_type});
  (void)args.insert({"x2_values", x2_values_type});
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);

  const std::set<TypePtr> valid_types1 = {kInt64};
  // Args x2_indices、x2_shape  type int64
  std::map<std::string, TypePtr> args1;
  (void)args1.insert({"x2_indices", x2_indices_type});
  (void)args1.insert({"x2_shape", x2_shape_type});
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args1, valid_types1, prim_name);

  return std::make_shared<Tuple>(std::vector<TypePtr>{kInt64, x1_type, kInt64});
}
}  // namespace

MIND_API_OPERATOR_IMPL(DenseToSparseSetOperation, BaseOperator);
AbstractBasePtr DenseToSparseSetOperationInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = DenseToSparseSetOperationInferType(primitive, input_args);
  auto infer_shape = DenseToSparseSetOperationInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGDenseToSparseSetOperationInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return DenseToSparseSetOperationInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return DenseToSparseSetOperationInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return DenseToSparseSetOperationInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(DenseToSparseSetOperation, prim::kPrimDenseToSparseSetOperation,
                                 AGDenseToSparseSetOperationInfer, false);
}  // namespace ops
}  // namespace mindspore
