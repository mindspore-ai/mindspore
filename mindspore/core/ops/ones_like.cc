/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "ops/ones_like.h"

#include <vector>
#include <memory>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr OnesLikeInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  return CheckAndConvertUtils::GetTensorInputShape(op_name, input_args, 0);
}

TypePtr OnesLikeInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto infer_type = input_args[0]->BuildType();
  auto valid_type = common_valid_types_with_complex_and_bool;
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", infer_type, valid_type, op_name);
  return infer_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(OnesLike, BaseOperator);
AbstractBasePtr OnesLikeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto infer_type = OnesLikeInferType(primitive, input_args);
  auto infer_shape = OnesLikeInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGOnesLikeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return OnesLikeInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return OnesLikeInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return OnesLikeInfer(engine, primitive, input_args);
  }

  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    if (input_args.empty()) {
      return nullptr;
    }

    auto op_name = primitive->name();
    auto shape_ptr = CheckAndConvertUtils::GetTensorInputShape(op_name, input_args, 0);
    auto shape_vec = shape_ptr->shape();
    if (IsDynamic(shape_vec)) {
      return nullptr;
    }

    auto infer_type = input_args[0]->BuildType();
    return TensorConstructUtils::CreateOnesTensor(infer_type, shape_vec);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(OnesLike, prim::kPrimOnesLike, AGOnesLikeInfer, false);
}  // namespace ops
}  // namespace mindspore
