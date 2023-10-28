/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "ops/add_ext.h"
#include <memory>
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

namespace mindspore {
namespace ops {
namespace {
BaseShapePtr AddInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack());
  auto y_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->GetShapeTrack());
  auto x_shape = x_shape_map[kShape];
  auto y_shape = y_shape_map[kShape];

  if (x_shape == y_shape) {
    return std::make_shared<abstract::Shape>(x_shape);
  }
  auto broadcast_shape = CalBroadCastShape(x_shape, y_shape, prim_name);
  auto output_shape_ptr = std::make_shared<abstract::Shape>(broadcast_shape);
  return output_shape_ptr;
}

TypePtr AddInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[0]->BuildType());
  (void)types.emplace("y", input_args[1]->BuildType());
  auto output_type = CheckAndConvertUtils::CheckMathBinaryOpTensorType(types, common_valid_types, prim_name);
  return output_type;
}
AbstractBasePtr AddInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (auto &input : input_args) {
    MS_EXCEPTION_IF_NULL(input);
  }
  constexpr int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto types = AddInferType(primitive, input_args);
  auto shapes = AddInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}
}  // namespace

MIND_API_OPERATOR_IMPL(AddExt, BaseOperator);
class MIND_API AGAddExtInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return AddInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return AddInferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return AddInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(AddExt, prim::kPrimAddExt, AGAddExtInfer, false);
}  // namespace ops
}  // namespace mindspore
