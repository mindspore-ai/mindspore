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
#include <map>
#include <string>
#include <memory>

#include "ops/less_equal.h"
#include "ops/op_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr LessEqualInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  return BroadCastInferShape(op_name, input_args);
}

TypePtr LessEqualInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim->name());
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[0]->BuildType());
  (void)types.emplace("y", input_args[1]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types_with_bool, prim->name());
  return std::make_shared<TensorType>(kBool);
}
}  // namespace

MIND_API_OPERATOR_IMPL(LessEqual, BaseOperator);
AbstractBasePtr LessEqualInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = LessEqualInferType(primitive, input_args);
  auto infer_shape = LessEqualInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGLessEqualInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LessEqualInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LessEqualInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return LessEqualInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(LessEqual, prim::kPrimLessEqual, AGLessEqualInfer, false);
}  // namespace ops
}  // namespace mindspore
