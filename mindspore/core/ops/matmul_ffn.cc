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

#include "ops/matmul_ffn.h"

#include <map>
#include <memory>
#include <set>
#include <string>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_ops.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr MatmulFfnInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->GetShape())[kShape];
  auto w1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->GetShape())[kShape];
  auto w2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->GetShape())[kShape];
  // (todo) check dynamic rank and dynamic shape
  if (IsDynamicRank(x_shape) || IsDynamicRank(w1_shape) || IsDynamicRank(w2_shape)) {
    MS_LOG(EXCEPTION) << "For " << op_name << ", dynamic rank is not supported";
  }
  const size_t x_rank = x_shape.size();
  const size_t w1_rank = w1_shape.size();
  const size_t w2_rank = w2_shape.size();
  MS_CHECK_VALUE(x_rank != 0 && x_rank == w1_rank && x_rank == w2_rank,
                 CheckAndConvertUtils::FormatCommMsg("For 'MatmulFfn', all inputs must have the same rank."));

  auto m = x_shape[0];
  auto k = x_shape[1];
  auto n0 = w1_shape[0];
  auto k0 = w1_shape[1];
  auto n1 = w2_shape[0];
  auto k1 = w2_shape[1];

  MS_CHECK_VALUE(k == k0 && k == k1, CheckAndConvertUtils::FormatCommMsg(
                                       "For 'MatmulFfn', the K axis of all inputs must have the same length."));

  ShapeVector output_1_shape = {m, n0};
  ShapeVector output_2_shape = {m, n1};
  std::vector<BaseShapePtr> shape_lists;
  (void)shape_lists.emplace_back(std::make_shared<abstract::TensorShape>(output_1_shape));
  (void)shape_lists.emplace_back(std::make_shared<abstract::TensorShape>(output_2_shape));
  return std::make_shared<abstract::TupleShape>(shape_lists);
}

TuplePtr MatmulFfnInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x_type = input_args[kInputIndex0]->GetType();
  // (todo) check dtype bf16 fp16 ...
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_type, x_type, x_type});
}

AbstractBasePtr MatmulFfnInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = MatmulFfnInferType(primitive, input_args);
  auto infer_shape = MatmulFfnInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
}  // namespace

MIND_API_OPERATOR_IMPL(MatmulFfn, BaseOperator);

// AG means auto generated
class MIND_API AGMatmulFfnInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MatmulFfnInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MatmulFfnInferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MatmulFfnInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MatmulFfn, prim::kPrimMatmulFfn, AGMatmulFfnInfer, false);
}  // namespace ops
}  // namespace mindspore
