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
#include "ops/grad/gather_d_grad_v2.h"

#include <memory>
#include <string>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::BaseShapePtr GatherDGradV2InferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const size_t op_input_num = 4;
  const size_t op_input_num_with_dim_attr = 3;
  auto input_num = input_args.size();
  MS_EXCEPTION_IF_CHECK_FAIL(input_num == op_input_num || input_num == op_input_num_with_dim_attr,
                             "GatherDGradV2's input size should be 3 or 4 but got " + std::to_string(input_num));
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  return input_args[kInputIndex0]->BuildShape();
}

TypePtr GatherDGradV2InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const size_t op_input_num = 4;
  const size_t op_input_num_with_dim_attr = 3;
  auto input_num = input_args.size();
  MS_EXCEPTION_IF_CHECK_FAIL(
    input_num == op_input_num || input_num == op_input_num_with_dim_attr,
    "GatherDGradV2's input size should be 3 or 4 but got " + std::to_string(input_args.size()));
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[kInputIndex0]->BuildType(), {kTensorType},
                                                   prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("index", input_args[input_num - kInputIndex2]->BuildType(),
                                                   {kInt32, kInt64}, prim_name);
  auto out_type = CheckAndConvertUtils::CheckTensorTypeValid("grad", input_args[input_num - kInputIndex1]->BuildType(),
                                                             {kTensorType}, prim_name);
  return out_type;
}
}  // namespace

AbstractBasePtr GatherDGradV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = GatherDGradV2InferType(primitive, input_args);
  auto infer_shape = GatherDGradV2InferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(GatherDGradV2, BaseOperator);

// AG means auto generated
class MIND_API AGGatherDGradV2Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return GatherDGradV2InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return GatherDGradV2InferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return GatherDGradV2Infer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(GatherDGradV2, prim::kPrimGatherDGradV2, AGGatherDGradV2Infer, false);
}  // namespace ops
}  // namespace mindspore
