/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "ops/zeros_like.h"

#include <vector>
#include <string>
#include <memory>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(ZerosLike, BaseOperator);
class ZerosLikeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    auto op_name = primitive->name();
    constexpr int64_t empty_tensor_num = 0;
    (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterThan, empty_tensor_num, op_name);
    auto input_shape_ptr = input_args[kInputIndex0]->BuildShape();
    auto input_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_shape_ptr);
    auto input_shape = input_shape_map[kShape];
    if (IsDynamicRank(input_shape)) {
      return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    }
    if (input_shape_ptr->IsDynamic()) {
      return input_shape_ptr->cast<abstract::ShapePtr>();
    }
    return CheckAndConvertUtils::GetTensorInputShape(op_name, input_args, 0);
  }
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto op_name = primitive->name();
    constexpr int64_t empty_tensor_num = 0;
    (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterThan, empty_tensor_num, op_name);
    MS_EXCEPTION_IF_NULL(input_args[0]);
    auto infer_type = input_args[0]->BuildType();
    auto valid_type = common_valid_types_with_complex_and_bool;
    (void)CheckAndConvertUtils::CheckTensorTypeValid("x", infer_type, valid_type, op_name);
    return infer_type;
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(ZerosLike, prim::kPrimZerosLike, ZerosLikeInfer, false);
}  // namespace ops
}  // namespace mindspore
