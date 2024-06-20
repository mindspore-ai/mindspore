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

#include "ops/uniform_int.h"
#include <memory>
#include <set>
#include <string>
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(UniformInt, BaseOperator);
void UniformInt::Init(int64_t seed, int64_t seed2) {
  this->set_seed(seed);
  this->set_seed2(seed2);
}

void UniformInt::set_seed(int64_t seed) { (void)this->AddAttr(kSeed, api::MakeValue(seed)); }

void UniformInt::set_seed2(int64_t seed2) { (void)this->AddAttr(kSeed2, api::MakeValue(seed2)); }

int64_t UniformInt::get_seed() const {
  auto value_ptr = GetAttr(kSeed);
  return GetValue<int64_t>(value_ptr);
}

int64_t UniformInt::get_seed2() const {
  auto value_ptr = GetAttr(kSeed2);
  return GetValue<int64_t>(value_ptr);
}

template <typename T>
BaseShapePtr TransUniformIntShapeValue(const T &shape_v, const PrimitivePtr &primitive) {
  if (!shape_v.has_value()) {
    ShapeVector dyn_output{abstract::TensorShape::kShapeRankAny};
    return std::make_shared<abstract::TensorShape>(dyn_output);
  }
  // shape_v will always have value.
  auto shape = shape_v.value();
  ShapeVector output_shape;
  for (size_t i = 0; i < shape_v->size(); i++) {
    if (shape.IsValueUnknown(i)) {
      output_shape.push_back(abstract::TensorShape::kShapeDimAny);
    } else {
      int64_t shape_i = shape[i];
      MS_CHECK_VALUE(shape_i >= 0, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                     "the " + std::to_string(i) + "th dimension of input shape", shape_i, kGreaterEqual,
                                     0, primitive));
      output_shape.push_back(shape_i);
    }
  }
  return std::make_shared<abstract::TensorShape>(output_shape);
}

BaseShapePtr UniformIntInferShape(const PrimitivePtr &primitive,
                                  const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto minval = CheckAndConvertUtils::CheckArgsType(op_name, input_args, kInputIndex1, kObjectTypeTensorType);
  MS_EXCEPTION_IF_NULL(minval);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("minval", minval->GetType(), {kInt32}, op_name);
  auto minval_shape = minval->GetShape();
  MS_EXCEPTION_IF_NULL(minval_shape);
  if (minval_shape->IsDimUnknown() || minval_shape->GetShapeVector().size() != 0) {
    MS_EXCEPTION(ValueError) << "For UniformInt, the min value should be a scalar tensor, while the shape is: "
                             << minval_shape->ToString();
  }
  auto maxval = CheckAndConvertUtils::CheckArgsType(op_name, input_args, kInputIndex2, kObjectTypeTensorType);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("maxval", maxval->GetType(), {kInt32}, op_name);
  auto maxval_shape = maxval->GetShape();
  MS_EXCEPTION_IF_NULL(maxval_shape);
  if (maxval_shape->IsDimUnknown() || minval_shape->GetShapeVector().size() != 0) {
    MS_EXCEPTION(ValueError) << "The max value should be a scalar tensor, while the shape is: "
                             << maxval_shape->ToString();
  }
  auto x = input_args[kInputIndex0];
  auto element_type = kNumberTypeInt64;
  if (CheckAndConvertUtils::IsTensor(x)) {
    auto tensor_type = x->GetType()->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type);
    element_type = tensor_type->element()->type_id();
  } else if (CheckAndConvertUtils::IsTuple(x)) {
    auto tuple_type = x->GetType()->cast<TuplePtr>();
    if (tuple_type->dynamic_len()) {
      element_type = tuple_type->dynamic_element_type()->type_id();
    } else {
      if (std::any_of(tuple_type->elements().cbegin(), tuple_type->elements().cend(),
                      [](const TypePtr &ele) { return ele->type_id() != kNumberTypeInt64; })) {
        element_type = kTypeUnknown;
      }
    }
  } else {
    MS_LOG(EXCEPTION) << "Unexpected type of input0:" << x->ToString() << " of op:" << primitive->ToString();
  }
  if (element_type == kNumberTypeInt64) {
    return TransUniformIntShapeValue(GetArrayValue<int64_t>(input_args[kInputIndex0]), primitive);
  } else if (element_type == kNumberTypeInt32) {
    return TransUniformIntShapeValue(GetArrayValue<int32_t>(input_args[kInputIndex0]), primitive);
  }
  MS_LOG(EXCEPTION) << "Unexpected element type of input0:" << x->ToString() << " of op:" << primitive->ToString();
}

class MIND_API UniformIntInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return UniformIntInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    const std::string &op_name = primitive->name();
    const int64_t kMinInputNum = 3;
    const int64_t kMaxInputNum = 5;
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual,
                                             kMinInputNum, op_name);
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kLessEqual, kMaxInputNum,
                                             op_name);
    return std::make_shared<TensorType>(kInt32);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {0}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(UniformInt, prim::kPrimUniformInt, UniformIntInfer, false);
}  // namespace ops
}  // namespace mindspore
