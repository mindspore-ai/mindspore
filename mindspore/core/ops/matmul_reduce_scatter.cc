/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/matmul_reduce_scatter.h"

#include <memory>
#include <vector>
#include <map>
#include <string>

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
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
namespace {
enum MatmulReduceScatterInputIndex : size_t {
  kMatmulReduceScatterInputX1Index = 0,
  kMatmulReduceScatterInputX2Index,
  kMatmulReduceScatterInputBiasIndex,
  kMatmulReduceScatterInputNum,
};
enum MatmulReduceScatterOutputIndex : size_t {
  kMatmulReduceScatterOutputYIndex = 0,
  kMatmulReduceScatterOutputNum,
};
constexpr size_t kX1X2Rank = 2;
constexpr size_t kBiasRank = 1;

abstract::TupleShapePtr MatmulReduceScatterInferShape(const PrimitivePtr &primitive,
                                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto is_trans_a = GetValue<bool>(primitive->GetAttr(kAttrIsTransA));
  auto is_trans_b = GetValue<bool>(primitive->GetAttr(kAttrIsTransB));
  auto x1_row = is_trans_a ? 1 : 0;
  auto x1_col = is_trans_a ? 0 : 1;
  auto x2_row = is_trans_b ? 1 : 0;
  auto x2_col = is_trans_b ? 0 : 1;
  auto x1_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kMatmulReduceScatterInputX1Index]->BuildShape())[kShape];
  auto x2_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kMatmulReduceScatterInputX2Index]->BuildShape())[kShape];
  if (x1_shape.size() != kX1X2Rank || x2_shape.size() != kX1X2Rank) {
    MS_LOG(EXCEPTION) << op_name << ": The rank both of x1 and x2 must be " << kX1X2Rank << ", but got "
                      << x1_shape.size() << " and " << x2_shape.size();
  }
  if (x1_shape[x1_col] != x2_shape[x2_row]) {
    MS_LOG(EXCEPTION) << op_name << ": The column of x1 and the row of x2 must be equal, but got " << x1_shape[x1_col]
                      << " and " << x2_shape[x2_row];
  }
  auto rank_size = GetValue<int64_t>(primitive->GetAttr(kAttrRankSize));
  if (rank_size <= 0 || x1_shape[x1_row] % rank_size != 0) {
    MS_LOG(EXCEPTION) << op_name << ": rank_size must a positive integer and be a divisor of x1.shape[" << x1_row
                      << "], but got " << rank_size << " and " << x1_shape[x1_row];
  }
  if (input_args[kMatmulReduceScatterInputBiasIndex]->BuildType()->type_id() != kMetaTypeNone) {
    auto bias_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
      input_args[kMatmulReduceScatterInputBiasIndex]->BuildShape())[kShape];
    ShapeVector expect_bias_shape{x2_shape[x2_col]};
    if (bias_shape != expect_bias_shape) {
      MS_LOG(EXCEPTION) << op_name << ": The shape of input `bias` must be " << expect_bias_shape << ", but got shape "
                        << bias_shape;
    }
  }
  abstract::BaseShapePtrList output_shape_ptr_list(kMatmulReduceScatterOutputNum);
  ShapeVector y_shape = {x1_shape[x1_row] / rank_size, x2_shape[x2_col]};
  output_shape_ptr_list[kMatmulReduceScatterOutputYIndex] = std::make_shared<abstract::Shape>(y_shape);
  return std::make_shared<abstract::TupleShape>(output_shape_ptr_list);
}

TuplePtr MatmulReduceScatterInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const std::set valid_types = {kFloat16, kBFloat16};
  auto op_name = primitive->name();
  std::map<std::string, TypePtr> types;
  // "x", "kernel_query", "kernel_key", "kernel_value", "gamma", " beta", "bias_query", "bias_key", "bias_value"
  (void)types.emplace("x1", input_args[kMatmulReduceScatterInputX1Index]->BuildType());
  (void)types.emplace("x2", input_args[kMatmulReduceScatterInputX2Index]->BuildType());
  if (input_args[kMatmulReduceScatterInputBiasIndex]->BuildType()->type_id() != kMetaTypeNone) {
    (void)types.emplace("bias", input_args[kMatmulReduceScatterInputBiasIndex]->BuildType());
  }
  auto type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  TypePtrList output_type_ptr_list(kMatmulReduceScatterOutputNum);
  output_type_ptr_list[kMatmulReduceScatterOutputYIndex] = type;
  return std::make_shared<Tuple>(output_type_ptr_list);
}
}  // namespace

AbstractBasePtr MatmulReduceScatterInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kMatmulReduceScatterInputNum, primitive->name());
  auto infer_type = MatmulReduceScatterInferType(primitive, input_args);
  auto infer_shape = MatmulReduceScatterInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(MatmulReduceScatter, BaseOperator);

// AG means auto generated
class MIND_API AGMatmulReduceScatterInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MatmulReduceScatterInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MatmulReduceScatterInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MatmulReduceScatterInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MatmulReduceScatter, prim::kPrimMatmulReduceScatter, AGMatmulReduceScatterInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
