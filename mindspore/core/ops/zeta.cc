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

#include <vector>
#include <map>
#include <set>
#include <string>
#include <memory>
#include <type_traits>
#include <utility>

#include "ops/zeta.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ZetaInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_shape_ptr = CheckAndConvertUtils::GetTensorInputShape("Zeta", input_args, 0);
  auto q_shape_ptr = CheckAndConvertUtils::GetTensorInputShape("Zeta", input_args, 1);
  auto x_shape = x_shape_ptr->shape();
  auto q_shape = q_shape_ptr->shape();
  // support dynamic rank
  if (IsDynamicRank(x_shape) || IsDynamicRank(q_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  // support dynamic rank
  if (IsDynamic(x_shape) || IsDynamic(q_shape)) {
    ShapeVector shape_out;
    for (size_t i = 0; i < x_shape.size(); ++i) {
      shape_out.push_back(abstract::Shape::kShapeDimAny);
    }
    return std::make_shared<abstract::Shape>(shape_out);
  }

  CheckAndConvertUtils::Check("input_x size", int64_t(x_shape.size()), kGreaterEqual, int64_t(q_shape.size()),
                              prim_name);
  if (x_shape.size() != 0 && x_shape[0] == 0) {
    MS_EXCEPTION(ValueError) << "For Zeta, the input_x must have value.";
  }
  if (q_shape.size() != 0 && q_shape[0] == 0) {
    MS_EXCEPTION(ValueError) << "For Zeta, the input_q must have value.";
  }
  if (*x_shape_ptr != *q_shape_ptr) {
    MS_EXCEPTION(ValueError) << primitive->name() << "Shape of x" << x_shape_ptr->ToString()
                             << " are not consistent with the shape q" << q_shape_ptr->ToString();
  }
  return x_shape_ptr;
}
TypePtr ZetaInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64};
  auto input_x = input_args[0]->BuildType();
  auto input_q = input_args[1]->BuildType();
  std::map<std::string, TypePtr> args_type;
  (void)args_type.insert(std::make_pair("x", input_x));
  (void)args_type.insert(std::make_pair("q", input_q));
  auto output_type = CheckAndConvertUtils::CheckTensorTypeSame(args_type, valid_types, primitive->name());
  return output_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(Zeta, BaseOperator);
AbstractBasePtr ZetaInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = ZetaInferType(primitive, input_args);
  auto infer_shape = ZetaInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGZetaInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ZetaInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ZetaInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ZetaInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Zeta, prim::kPrimZeta, AGZetaInfer, false);
}  // namespace ops
}  // namespace mindspore
