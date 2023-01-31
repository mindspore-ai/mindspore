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
#include <string>
#include <memory>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

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

BaseShapePtr UniformIntInferShape(const PrimitivePtr &primitive,
                                  const std::vector<abstract::AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  abstract::AbstractTensorPtr minval = abstract::CheckArg<abstract::AbstractTensor>(op_name, input_args, kInputIndex1);
  MS_EXCEPTION_IF_NULL(minval);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("minval", minval->BuildType(), {kInt32}, op_name);
  abstract::ShapePtr minval_shape = minval->shape();
  MS_EXCEPTION_IF_NULL(minval_shape);
  if (minval_shape->IsDimUnknown() || minval_shape->shape().size() != 0) {
    MS_EXCEPTION(ValueError) << "The min value should be a scalar tensor, while the shape is: "
                             << minval_shape->ToString();
  }
  abstract::AbstractTensorPtr maxval = abstract::CheckArg<abstract::AbstractTensor>(op_name, input_args, kInputIndex2);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("maxval", maxval->BuildType(), {kInt32}, op_name);
  abstract::ShapePtr maxval_shape = maxval->shape();
  MS_EXCEPTION_IF_NULL(maxval_shape);
  if (maxval_shape->IsDimUnknown() || maxval_shape->shape().size() != 0) {
    MS_EXCEPTION(ValueError) << "The max value should be a scalar tensor, while the shape is: "
                             << maxval_shape->ToString();
  }

  ShapeVector shape;
  abstract::ShapePtr output_shape;

  auto shape_value = input_args[kInputIndex0]->BuildValue();
  if (IsValueKnown(shape_value)) {
    shape = shape_value->isa<tensor::Tensor>()
              ? CheckAndConvertUtils::CheckTensorIntValue("input[shape]", shape_value, op_name)
              : CheckAndConvertUtils::CheckTupleInt("input[shape]", shape_value, op_name);
    output_shape = std::make_shared<abstract::Shape>(shape);
  } else {
    // ToSupport Dynamic
    shape = {-2};  // unknown dimension.
    output_shape = std::make_shared<abstract::Shape>(shape);
  }
  return output_shape;
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
    return kInt32;
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {0}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(UniformInt, prim::kPrimUniformInt, UniformIntInfer, false);
}  // namespace ops
}  // namespace mindspore
