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

#include <set>
#include "ops/in_top_k.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
constexpr auto kK = "k";
MIND_API_OPERATOR_IMPL(InTopK, BaseOperator);
void InTopK::Init(const int64_t k) { this->set_k(k); }
void InTopK::set_k(const int64_t k) { (void)this->AddAttr(kK, api::MakeValue(k)); }

int64_t InTopK::get_k() const {
  auto value_ptr = this->GetAttr(kK);
  return GetValue<int64_t>(value_ptr);
}

class InTopKInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    const int64_t kInputx1ShapeSize = 2;
    const int64_t kInputx2ShapeSize = 1;
    auto prim_name = primitive->name();
    auto x1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
    auto x2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
    if (IsDynamicRank(x1_shape) || IsDynamicRank(x2_shape)) {
      return std::make_shared<abstract::Shape>(std::vector<int64_t>{UNKNOWN_RANK});
    }
    (void)CheckAndConvertUtils::CheckInteger("input x1 rank", SizeToLong(x1_shape.size()), kEqual, kInputx1ShapeSize,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckInteger("input x2 rank", SizeToLong(x2_shape.size()), kEqual, kInputx2ShapeSize,
                                             prim_name);
    if (x2_shape[0] != x1_shape[0]) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', the size of x2 should be equal to x1's first diemnsion, but got x1 shape: "
                               << x1_shape << ", x2 shape: " << x2_shape;
    }
    auto x2 = input_args[kInputIndex1]->BuildShape();
    MS_EXCEPTION_IF_NULL(x2);
    auto shape_element = x2->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape_element);
    return shape_element;
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto prim_name = primitive->name();
    const std::set<TypePtr> x1_valid_types = {kFloat16, kFloat32};
    (void)CheckAndConvertUtils::CheckTensorTypeValid("x1", input_args[kInputIndex0]->BuildType(), x1_valid_types,
                                                     prim_name);
    const std::set<TypePtr> x2_valid_types = {kInt32};
    (void)CheckAndConvertUtils::CheckTensorTypeValid("x2", input_args[kInputIndex1]->BuildType(), x2_valid_types,
                                                     prim_name);
    return std::make_shared<TensorType>(kBool);
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(InTopK, prim::kPrimInTopK, InTopKInfer, false);
}  // namespace ops
}  // namespace mindspore
