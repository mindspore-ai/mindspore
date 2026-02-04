/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "infer/quant_batch_matmul_allreduce.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ops_utils/op_constants.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/type_id.h"
#include "mindapi/ir/value.h"
#include "mindapi/helper.h"
#include "primitive/math_ops.h"
#include "primitive/lite_ops.h"
#include "primitive/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "utils/ms_context.h"
#include "primitive/auto_generate/gen_ops_primitive_q.h"

namespace mindspore {
namespace ops {
abstract::TupleShapePtr QuantBatchMatmulAllReduceInferShape(const PrimitivePtr &primitive,
                                                            const std::vector<AbstractBasePtr> &input_args) {
  const std::string op_name = primitive->name();
  auto x1 = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 0, kObjectTypeTensorType);
  auto x2 = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 1, kObjectTypeTensorType);
  const auto &x1_shp = x1->GetShape()->GetShapeVector();
  const auto &x2_shp = x2->GetShape()->GetShapeVector();

  bool transpose_a = GetValue<bool>(primitive->GetAttr(kAttrIsTransA));
  bool transpose_b = GetValue<bool>(primitive->GetAttr(kAttrIsTransB));

  if (IsDynamicRank(x1_shp) || IsDynamicRank(x2_shp)) {
    std::vector<BaseShapePtr> output_shape_ptr_list;
    output_shape_ptr_list.emplace_back(
      std::make_shared<abstract::TensorShape>(ShapeVector{abstract::Shape::kShapeRankAny}));
    return std::make_shared<abstract::TupleShape>(output_shape_ptr_list);
  }

  const size_t SHAPE_SIZE = 2;
  if (x1_shp.size() != SHAPE_SIZE || x2_shp.size() != SHAPE_SIZE) {
    MS_EXCEPTION(ValueError) << "QuantBatchMatMulAllReduce inputs should have the same dimension size and equal to 2.";
  }

  auto x1_col = x1_shp[(transpose_a ? 0 : 1)];
  auto x2_row = x2_shp[(transpose_b ? 1 : 0)];
  if (x1_col != x2_row && x1_col >= 0 && x2_row >= 0) {
    MS_EXCEPTION(ValueError) << "For 'QuantBatchMatMulAllReduce' the input dimensions must be equal, but got 'x1_col': "
                             << x1_col << " and 'x2_row': " << x2_row << ".";
  }

  ShapeVector ret_shape;
  auto make_shape = [&transpose_a, &transpose_b](ShapeVector &output, const ShapeVector x1_shp,
                                                 const ShapeVector x2_shp) -> void {
    if (!x1_shp.empty() && !x2_shp.empty()) {
      output.push_back(x1_shp[(transpose_a ? 1 : 0)]);
      output.push_back(x2_shp[(transpose_b ? 0 : 1)]);
    }
    return;
  };
  make_shape(ret_shape, x1_shp, x2_shp);
  std::vector<BaseShapePtr> output_shape_ptr_list;
  output_shape_ptr_list.emplace_back(std::make_shared<abstract::TensorShape>(ret_shape));
  return std::make_shared<abstract::TupleShape>(output_shape_ptr_list);
}

TuplePtr QuantBatchMatmulAllReduceInferType(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
  auto x1_type = input_args[kIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(input_args[kIndex1]);
  auto x2_type = input_args[kIndex1]->GetType();
  MS_EXCEPTION_IF_NULL(input_args[kIndex4]);
  auto scale_type = input_args[kIndex4]->GetType();
  MS_EXCEPTION_IF_NULL(input_args[kIndex5]);
  auto pertoken_scale_type = input_args[kIndex5]->GetType();

  std::map<std::string, TypePtr> types;
  (void)types.emplace("x1", x1_type);
  (void)types.emplace("x2", x2_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, {kInt8}, primitive->name());
  types.clear();
  (void)types.emplace("scale", scale_type);
  (void)types.emplace("pertoken_scale", pertoken_scale_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, {kFloat32}, primitive->name());
  ValuePtr dtype_ptr = input_args[kIndex6]->GetValue();
  auto dtype = GetValue<int64_t>(dtype_ptr);
  if (dtype != TypeId::kNumberTypeFloat16 && dtype != TypeId::kNumberTypeBFloat16) {
    MS_EXCEPTION(TypeError) << "QuantBatchMatMulAllReduce's output dtype only support fp16 or bf16.";
  }
  return std::make_shared<Tuple>(std::vector<TypePtr>{dtype == kNumberTypeFloat16 ? kFloat16 : kBFloat16});
}

AbstractBasePtr QuantBatchMatmulAllReduceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 6;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputsNum, primitive->name());
  auto infer_type = QuantBatchMatmulAllReduceInferType(primitive, input_args);
  auto infer_shape = QuantBatchMatmulAllReduceInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(QuantBatchMatmulAllReduce, BaseOperator);
class OPS_API AGQuantBatchMatmulAllReduceInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return QuantBatchMatmulAllReduceInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return QuantBatchMatmulAllReduceInferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return QuantBatchMatmulAllReduceInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(QuantBatchMatmulAllReduce, prim::kPrimQuantBatchMatmulAllReduce,
                                 AGQuantBatchMatmulAllReduceInfer, false);
}  // namespace ops
}  // namespace mindspore
