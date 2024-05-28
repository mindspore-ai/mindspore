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
  auto w_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->GetShape())[kShape];
  // (todo) check dynamic rank and dynamic shape
  if (IsDynamicRank(x_shape) || IsDynamicRank(w_shape)) {
    MS_LOG(EXCEPTION) << "For " << op_name << ", dynamic rank is not supported";
  }
  const size_t x_rank = x_shape.size();
  const size_t w_rank = w_shape.size();
  MS_CHECK_VALUE(x_rank != 0 && x_rank == w_rank,
                 CheckAndConvertUtils::FormatCommMsg("For 'MatmulQkv', all inputs must have the same rank."));

  auto m = x_shape[0];
  auto k = x_shape[1];
  auto k0 = w_shape[1];
  MS_CHECK_VALUE(k == k0, CheckAndConvertUtils::FormatCommMsg(
                            "For 'MatmulQkv', the K axis of all inputs must have the same length."));

  MS_CHECK_VALUE(primitive->HasAttr("n_lens"),
                 CheckAndConvertUtils::FormatCommMsg("For 'MatmulQkv', op must have attr 'n_lens'."));
  std::vector<int64_t> n_len_list = GetValue<std::vector<int64_t>>(primitive->GetAttr("n_lens"));

  ShapeVector output_q_shape = {m, n_len_list[0]};
  ShapeVector output_k_shape = {m, n_len_list[1]};
  ShapeVector output_v_shape = {m, n_len_list[2]};
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
  MS_CHECK_VALUE(primitive->HasAttr("is_fixed_weight"),
                 CheckAndConvertUtils::FormatCommMsg("For 'MatmulQkv', op must have attr 'is_fixed_weight'."));
  auto is_fixed_weight = GetValue<bool>(primitive->GetAttr("is_fixed_weight"));
  if (is_fixed_weight) {
    const auto input_num = 2;
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  } else {
    // two cases: 1 input with 2 weights or 3 weights
    const auto qkv_num = 4;
    const auto ffn_num = 3;
    MS_CHECK_VALUE(input_args.size() == ffn_num || input_args.size() == qkv_num,
                   CheckAndConvertUtils::FormatCommMsg(
                     "For 'MatmulQkv' in multi weight mode, number of weights must be 2 or 3, but got " +
                     std::to_string(static_cast<int>(input_args.size() - 1)) + "."));
  }

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
