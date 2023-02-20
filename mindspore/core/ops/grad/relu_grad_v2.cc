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
#include "ops/grad/relu_grad_v2.h"

#include <string>
#include <map>
#include <vector>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(ReluGradV2, BaseOperator);
class ReluGradV2Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    auto gradient_shape_ptr = input_args[kGradientIndex]->BuildShape();
    auto gradient_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(gradient_shape_ptr);
    auto gradient_input_shape = gradient_shape_map[kShape];
    auto mask_shape_ptr = input_args[kMaskIndex]->BuildShape();
    auto mask_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(mask_shape_ptr);
    auto mask_input_shape = mask_shape_map[kShape];
    if (IsDynamicRank(gradient_input_shape) || IsDynamicRank(mask_input_shape)) {
      return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    }
    if (mask_input_shape.size() < kReluGradV2GradientDims) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', The 'mask' dims must be greater than 4,but got " +
                                    std::to_string(mask_input_shape.size()) + "-D tensor";
    }
    if (gradient_input_shape.size() < kReluGradV2GradientDims) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', The dims of 'gradient' must be greater than 4,but got a " +
                                    std::to_string(gradient_input_shape.size()) + "-D tensor";
    }
    // Dynamic Shape
    if (gradient_shape_ptr->IsDynamic() || mask_shape_ptr->IsDynamic()) {
      ShapeVector shape_out;
      for (size_t i = 0; i < gradient_input_shape.size(); ++i) {
        shape_out.push_back(abstract::Shape::kShapeDimAny);
      }
      return std::make_shared<abstract::Shape>(shape_out);
    }

    return gradient_shape_ptr;
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    MS_EXCEPTION_IF_NULL(input_args[kGradientIndex]);
    auto gradient_type = input_args[kGradientIndex]->BuildType();
    MS_EXCEPTION_IF_NULL(gradient_type);
    if (!gradient_type->isa<TensorType>()) {
      MS_EXCEPTION(TypeError) << "The " << prim_name << "'s "
                              << " input must be tensor type but got " << gradient_type->ToString();
    }
    return gradient_type;
  }

 private:
  const size_t kReluGradV2InputNum = 2;
  const size_t kGradientIndex = 0;
  const size_t kMaskIndex = 1;
  const size_t kReluGradV2GradientDims = 4;
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ReluGradV2, prim::kPrimReluGradV2, ReluGradV2Infer, false);
}  // namespace ops
}  // namespace mindspore
