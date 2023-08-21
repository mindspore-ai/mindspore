/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/npu_antiquant.h"
#include <set>
#include <string>
#include <vector>
#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "mindspore/core/ops/other_ops.h"

namespace mindspore {
namespace ops {
namespace {
BaseShapePtr NPUAntiQuantInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr int64_t kNumber1 = 1;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, kNumber1, prim_name);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
  auto x = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto shape_element = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr NPUAntiQuantInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  std::map<std::string, TypePtr> types;
  std::set<TypePtr> valid_types = {kInt8};
  TypePtr input_x_type = input_args[0]->BuildType();
  (void)types.emplace("input_x", input_x_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  auto dst_type = primitive->GetAttr(kOutputDType);
  if ((dst_type == nullptr) || (!dst_type->isa<Type>())) {
    MS_EXCEPTION(TypeError) << "Invalid dtype";
  }
  return dst_type->cast<TypePtr>();
}
}  // namespace

AbstractBasePtr NPUAntiQuantInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (auto &input : input_args) {
    MS_EXCEPTION_IF_NULL(input);
  }
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto types = NPUAntiQuantInferType(primitive, input_args);
  auto shapes = NPUAntiQuantInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

MIND_API_OPERATOR_IMPL(NPUAntiQuant, BaseOperator);
class MIND_API AGNPUAntiQuantInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return NPUAntiQuantInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return NPUAntiQuantInferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return NPUAntiQuantInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(NPUAntiQuant, prim::kPrimNPUAntiQuant, AGNPUAntiQuantInfer, false);
}  // namespace ops
}  // namespace mindspore
