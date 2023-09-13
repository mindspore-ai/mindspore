/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include "ops/bn_infer.h"
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
// here A for Ascend, as it will call BNInfer instead of Batchnorm::Infer func
// when running batchnorm infer on Ascend
class ABNInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    auto x_shape_ptr = input_args[kInputIndex0]->BuildShape();
    auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
    auto scale_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
    if (!(IsDynamic(x_shape) || IsDynamic(scale_shape))) {
      auto scale_channel = scale_shape.size() == kInputIndex1 ? scale_shape[kInputIndex0] : scale_shape[kInputIndex1];
      auto x_channel = x_shape[kInputIndex1];
      if (scale_channel != x_channel) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name
                                 << "', 'scale_dim0' and input channel should be equal, but got " << scale_channel
                                 << " and " << x_channel << ".";
      }
    }
    return x_shape_ptr;
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
    auto x_type = input_args[0]->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", x_type, valid_types, prim->name());

    return x_type;
  }
};
MIND_API_OPERATOR_IMPL(BNInfer, BaseOperator);
abstract::AbstractBasePtr BNInferFunc(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  ABNInfer bn;
  auto type = bn.InferType(primitive, input_args);
  auto shape = bn.InferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

REGISTER_PRIMITIVE_OP_INFER_IMPL(BNInfer, prim::kPrimBNInfer, ABNInfer, false);
}  // namespace ops
}  // namespace mindspore
