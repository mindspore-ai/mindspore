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

#include "ops/matmul_qkv.h"

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
abstract::TupleShapePtr MatmulQkvInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->GetShape())[kShape];
  auto wq_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->GetShape())[kShape];
  auto wk_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->GetShape())[kShape];
  auto wv_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->GetShape())[kShape];
  // (todo) check dynamic rank and dynamic shape
  if (IsDynamicRank(x_shape) || IsDynamicRank(wq_shape) || IsDynamicRank(wk_shape) || IsDynamicRank(wv_shape)) {
    MS_LOG(EXCEPTION) << "For " << op_name << ", dynamic rank is not supported";
  }
  const size_t x_rank = x_shape.size();
  const size_t wq_rank = wq_shape.size();
  const size_t wk_rank = wk_shape.size();
  const size_t wv_rank = wv_shape.size();
  MS_CHECK_VALUE(x_rank != 0 && x_rank == wq_rank && x_rank == wk_rank && x_rank == wv_rank,
                 CheckAndConvertUtils::FormatCommMsg("For 'MatmulQkv', all inputs must have the same rank."));

  auto m = x_shape[0];
  auto k = x_shape[1];
  auto n0 = wq_shape[0];
  auto k0 = wq_shape[1];
  auto n1 = wk_shape[0];
  auto k1 = wk_shape[1];
  auto n2 = wv_shape[0];
  auto k2 = wv_shape[1];
  MS_CHECK_VALUE(
    k == k0 && k == k1 && k == k2,
    CheckAndConvertUtils::FormatCommMsg("For 'MatmulQkv', the K axis of all inputs must have the same length."));

  ShapeVector output_q_shape = {m, n0};
  ShapeVector output_k_shape = {m, n1};
  ShapeVector output_v_shape = {m, n2};
  std::vector<BaseShapePtr> shape_lists;
  (void)shape_lists.emplace_back(std::make_shared<abstract::TensorShape>(output_q_shape));
  (void)shape_lists.emplace_back(std::make_shared<abstract::TensorShape>(output_k_shape));
  (void)shape_lists.emplace_back(std::make_shared<abstract::TensorShape>(output_v_shape));
  return std::make_shared<abstract::TupleShape>(shape_lists);
}

TuplePtr MatmulQkvInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x_type = input_args[kInputIndex0]->GetType();
  // (todo) check dtype bf16 fp16 ...
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_type, x_type, x_type});
}

AbstractBasePtr MatmulQkvInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = MatmulQkvInferType(primitive, input_args);
  auto infer_shape = MatmulQkvInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
}  // namespace

MIND_API_OPERATOR_IMPL(MatmulQkv, BaseOperator);

// AG means auto generated
class MIND_API AGMatmulQkvInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MatmulQkvInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MatmulQkvInferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MatmulQkvInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MatmulQkv, prim::kPrimMatmulQkv, AGMatmulQkvInfer, false);
}  // namespace ops
}  // namespace mindspore
