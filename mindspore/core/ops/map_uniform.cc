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

#include "ops/map_uniform.h"

#include <set>

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
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kMapUniformInputsNum = 3;
constexpr size_t kHashMapDim = 2;
abstract::ShapePtr MapUniformInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  return input_args[kInputIndex0]->BuildShape()->cast<abstract::ShapePtr>();
}

TypePtr MapUniformInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  MS_EXCEPTION_IF_NULL(prim);
  auto input_type = input_args[kInputIndex0]->BuildType();
  auto per_group_size_type = input_args[kInputIndex1]->BuildType();
  auto group_num_type = input_args[kInputIndex2]->BuildType();

  const std::set<TypePtr> input_valid_types = {kInt64, kInt32, kInt16, kInt8};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input", input_type, input_valid_types, prim_name);

  const std::set<TypePtr> group_valid_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTypeValid("per_group_size", per_group_size_type, group_valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTypeValid("group_num", group_num_type, group_valid_types, prim_name);

  return input_type;
}
}  // namespace

AbstractBasePtr MapUniformInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kMapUniformInputsNum, primitive->name());
  auto infer_type = MapUniformInferType(primitive, input_args);
  auto infer_shape = MapUniformInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(MapUniform, BaseOperator);

// AG means auto generated
class MIND_API AGMapUniformInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MapUniformInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MapUniformInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MapUniformInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MapUniform, prim::kPrimMapUniform, AGMapUniformInfer, false);
}  // namespace ops
}  // namespace mindspore
