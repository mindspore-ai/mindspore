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
#include <iterator>
#include <memory>
#include <set>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
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
    size_t batch_rank = 0;
    if (primitive->HasAttr(kBatchRank)) {
      auto value_ptr = primitive->GetAttr(kBatchRank);
      batch_rank = GetValue<int64_t>(value_ptr);
    }
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
    ShapeVector count_shape;
    (void)copy(shape_vec.begin(), shape_vec.begin() + batch_rank, std::back_inserter(count_shape));
    count_shape.push_back(count_value);
    auto count_shape_ptr = std::make_shared<abstract::Shape>(count_shape);

    if (IsDynamicRank(shape_vec)) {
      auto first_output_shape_ptr =
        std::make_shared<abstract::Shape>(ShapeVector({count_value, abstract::Shape::kShapeDimAny}));
      std::make_shared<abstract::TupleShape>(
        std::vector<abstract::BaseShapePtr>{first_output_shape_ptr, count_shape_ptr});
    }

    auto shape_rank = shape_vec.size();
    if (shape_rank < kDim1 + batch_rank || shape_rank > kDim5 + batch_rank) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', input[0] rank should be between 1 and 5, but got:" << shape_rank;
    }

    ShapeVector index_shape;
    (void)copy(shape_vec.begin(), shape_vec.begin() + batch_rank, std::back_inserter(index_shape));
    index_shape.push_back(count_value);
    index_shape.push_back(static_cast<int64_t>(shape_rank - batch_rank));
    auto first_output_shape_ptr = std::make_shared<abstract::Shape>(index_shape);
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
