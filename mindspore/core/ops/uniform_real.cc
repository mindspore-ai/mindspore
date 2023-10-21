/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/uniform_real.h"
#include <memory>
#include <set>
#include <string>
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(UniformReal, BaseOperator);
MIND_API_OPERATOR_IMPL(CudnnUniformReal, BaseOperator);
void UniformReal::Init(int64_t seed, int64_t seed2) {
  this->set_seed(seed);
  this->set_seed2(seed2);
}

void UniformReal::set_seed(int64_t seed) { (void)this->AddAttr(kSeed, api::MakeValue(seed)); }

void UniformReal::set_seed2(int64_t seed2) { (void)this->AddAttr(kSeed2, api::MakeValue(seed2)); }

int64_t UniformReal::get_seed() const {
  auto value_ptr = GetAttr(kSeed);
  return GetValue<int64_t>(value_ptr);
}

int64_t UniformReal::get_seed2() const {
  auto value_ptr = GetAttr(kSeed2);
  return GetValue<int64_t>(value_ptr);
}

BaseShapePtr UniformRealInferShape(const PrimitivePtr &primitive,
                                   const std::vector<abstract::AbstractBasePtr> &input_args) {
  ShapeVector shape;
  abstract::ShapePtr output_shape;
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto shape_value = input_args[kInputIndex0]->BuildValue();
  if (IsValueKnown(shape_value)) {
    shape = shape_value->isa<tensor::Tensor>()
              ? CheckAndConvertUtils::CheckTensorIntValue("input[shape]", shape_value, primitive->name())
              : CheckAndConvertUtils::CheckTupleInt("input[shape]", shape_value, primitive->name());
    output_shape = std::make_shared<abstract::Shape>(shape);
  } else {
    // ToSupport Dynamic
    shape = {-2};  // unknown dimension.
    output_shape = std::make_shared<abstract::Shape>(shape);
  }
  return output_shape;
}

class MIND_API UniformRealInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return UniformRealInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    const std::string &op_name = primitive->name();
    const int64_t kMinInputNum = 1;
    const int64_t kMaxInputNum = 3;
    // Check Input
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual,
                                             kMinInputNum, op_name);
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kLessEqual, kMaxInputNum,
                                             op_name);
    return std::make_shared<TensorType>(kFloat32);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {0}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(UniformReal, prim::kPrimUniformReal, UniformRealInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(CudnnUniformReal, prim::kPrimCudnnUniformReal, UniformRealInfer, false);
}  // namespace ops
}  // namespace mindspore
