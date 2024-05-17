/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/base/type_id.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/lite_ops.h"
#include "ops/matmul_allreduce.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
abstract::TupleShapePtr MatMulAllReduceInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  const std::string op_name = primitive->name();
  auto x = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 0, kObjectTypeTensorType);

  auto y = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 1, kObjectTypeTensorType);
  const auto &x_shp = x->GetShape()->GetShapeVector();
  const auto &y_shp = y->GetShape()->GetShapeVector();

  ValuePtr transpose_a_ptr = primitive->GetAttr("transpose_a");
  ValuePtr transpose_b_ptr = primitive->GetAttr("transpose_b");
  bool transpose_a = GetValue<bool>(transpose_a_ptr);
  bool transpose_b = GetValue<bool>(transpose_b_ptr);

  if (IsDynamicRank(x_shp) || IsDynamicRank(y_shp)) {
    ShapeVector ret_shape{abstract::Shape::kShapeRankAny};
    std::vector<BaseShapePtr> output_shape_ptr_list;
    output_shape_ptr_list.emplace_back(std::make_shared<abstract::TensorShape>(ret_shape));
    return std::make_shared<abstract::TupleShape>(output_shape_ptr_list);
  }

  if (x_shp.size() == 1 && y_shp.size() == 1 && x_shp[0] == 0 && y_shp[0] == 0) {
    ShapeVector ret_shape;
    std::vector<BaseShapePtr> output_shape_ptr_list;
    output_shape_ptr_list.emplace_back(std::make_shared<abstract::TensorShape>(ret_shape));
    return std::make_shared<abstract::TupleShape>(output_shape_ptr_list);
  }

  const size_t SHAPE_SIZE = 2;
  if (x_shp.size() != SHAPE_SIZE || y_shp.size() != SHAPE_SIZE) {
    MS_EXCEPTION(ValueError) << "MatMulAllReduce inputs should have the same dimension size and equal to 2.";
  }
  auto x_col = x_shp[(transpose_a ? 0 : 1)];
  auto y_row = y_shp[(transpose_b ? 1 : 0)];
  if (x_col != y_row && x_col >= 0 && y_row >= 0) {
    MS_EXCEPTION(ValueError) << "For 'MatMulAllReduce' the input dimensions must be equal, but got 'x1_col': " << x_col
                             << " and 'x2_row': " << y_row << ".";
  }

  ShapeVector ret_shape;
  auto make_shape = [&transpose_a, &transpose_b](ShapeVector &output, const ShapeVector xshp,
                                                 const ShapeVector yshp) -> void {
    if (!xshp.empty() && !yshp.empty()) {
      output.push_back(xshp[(transpose_a ? 1 : 0)]);
      output.push_back(yshp[(transpose_b ? 0 : 1)]);
    }
    return;
  };
  make_shape(ret_shape, x_shp, y_shp);
  std::vector<BaseShapePtr> output_shape_ptr_list;
  output_shape_ptr_list.emplace_back(std::make_shared<abstract::TensorShape>(ret_shape));
  return std::make_shared<abstract::TupleShape>(output_shape_ptr_list);
}

TuplePtr MatMulAllReduceInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto x = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 0, kObjectTypeTensorType);
  auto y = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 1, kObjectTypeTensorType);

  auto x_tensor_type = x->GetType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(x_tensor_type);
  auto y_tensor_type = y->GetType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(y_tensor_type);
  TypePtr x_type = x_tensor_type->element();
  TypePtr y_type = y_tensor_type->element();

  if (x_type->type_id() != y_type->type_id()) {
    MS_EXCEPTION(TypeError) << "For '" << op_name
                            << "', the type of 'x2' should be same as 'x1', but got 'x1' with type Tensor["
                            << x_type->ToString() << "] and 'x2' with type Tensor[" << y_type->ToString() << "].";
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[kInputIndex0]->GetType());
  (void)types.emplace("y", input_args[kInputIndex1]->GetType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_type});
}

AbstractBasePtr MatMulAllReduceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputsNum, primitive->name());
  auto infer_type = MatMulAllReduceInferType(primitive, input_args);
  auto infer_shape = MatMulAllReduceInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(MatMulAllReduce, BaseOperator);
class MIND_API AGMatMulAllReduceInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MatMulAllReduceInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MatMulAllReduceInferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MatMulAllReduceInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MatMulAllReduce, prim::kPrimMatMulAllReduce, AGMatMulAllReduceInfer, false);
}  // namespace ops
}  // namespace mindspore
