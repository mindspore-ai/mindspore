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
#include "ops/random_choice_with_mask.h"
#include <string>
#include <algorithm>
#include <vector>
#include "ops/op_utils.h"
#include "abstract/param_validator.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void RandomChoiceWithMask::set_seed(const int64_t seed) { (void)this->AddAttr("seed", api::MakeValue(seed)); }

void RandomChoiceWithMask::set_seed2(const int64_t seed2) { (void)this->AddAttr("seed2", api::MakeValue(seed2)); }

void RandomChoiceWithMask::set_count(const int64_t count) { (void)this->AddAttr("count", api::MakeValue(count)); }

int64_t RandomChoiceWithMask::get_seed() const {
  auto value_ptr = GetAttr("seed");
  return GetValue<int64_t>(value_ptr);
}

int64_t RandomChoiceWithMask::get_seed2() const {
  auto value_ptr = GetAttr("seed2");
  return GetValue<int64_t>(value_ptr);
}

int64_t RandomChoiceWithMask::get_count() const {
  auto value_ptr = GetAttr("count");
  return GetValue<int64_t>(value_ptr);
}

MIND_API_OPERATOR_IMPL(RandomChoiceWithMask, BaseOperator);
class RandomChoiceWithMaskInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    const int64_t kRandomChoiceWithMaskInputsNum = 1;
    const int64_t input_num = kRandomChoiceWithMaskInputsNum;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                             prim_name);
    MS_EXCEPTION_IF_NULL(input_args.front());
    auto input_x_shape_ptr = input_args[kInputIndex0]->BuildShape();
    MS_EXCEPTION_IF_NULL(input_x_shape_ptr);
    if (!input_x_shape_ptr->isa<abstract::Shape>()) {
      MS_LOG(EXCEPTION) << "For '" << primitive->name()
                        << "', input[0] should be a Tensor, but got:" << input_x_shape_ptr->ToString();
    }
    const auto &shape_vec = input_x_shape_ptr->cast<abstract::ShapePtr>()->shape();

    auto value_ptr = primitive->GetAttr("count");
    auto count_value = GetValue<int64_t>(value_ptr);
    auto count_shape_ptr = std::make_shared<abstract::Shape>(std::vector<int64_t>{count_value});

    if (IsDynamicRank(shape_vec)) {
      auto first_output_shape_ptr =
        std::make_shared<abstract::Shape>(ShapeVector({count_value, abstract::Shape::kShapeDimAny}));
      std::make_shared<abstract::TupleShape>(
        std::vector<abstract::BaseShapePtr>{first_output_shape_ptr, count_shape_ptr});
    }

    auto shape_rank = shape_vec.size();
    if (shape_rank < kDim1 || shape_rank > kDim5) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', input[0] rank should be between 1 and 5, but got:" << shape_rank;
    }

    auto first_output_shape_ptr =
      std::make_shared<abstract::Shape>(ShapeVector({count_value, static_cast<int64_t>(shape_rank)}));
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{first_output_shape_ptr, count_shape_ptr});
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    const int64_t kRandomChoiceWithMaskInputsNum = 1;
    const int64_t input_num = kRandomChoiceWithMaskInputsNum;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                             prim_name);
    auto x_type = input_args[kInputIndex0]->BuildType();
    MS_EXCEPTION_IF_NULL(x_type);
    if (!x_type->isa<TensorType>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', input must be a Tensor, but got: " << x_type->ToString()
                              << ".";
    }

    const std::set<TypePtr> valid1_types = {kBool};
    (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", x_type, valid1_types, prim_name);
    return std::make_shared<Tuple>(std::vector<TypePtr>{kInt32, kBool});
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(RandomChoiceWithMask, prim::kPrimRandomChoiceWithMask, RandomChoiceWithMaskInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
