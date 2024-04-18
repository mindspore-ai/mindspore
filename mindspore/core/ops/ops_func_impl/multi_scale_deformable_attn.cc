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
#include <set>
#include <string>
#include "abstract/ops/primitive_infer_map.h"
#include "ops/nn_ops.h"
#include "utils/check_convert_utils.h"
#include "ops/primitive_c.h"
#include "mindapi/src/helper.h"
#include "ops/ops_func_impl/multi_scale_deformable_attn.h"
#include "ops/auto_generate/gen_lite_ops.h"

namespace mindspore {
namespace ops {

enum MultiScaleDeformableAttnInputIndex : size_t {
  kMultiScaleDeformableAttnInputValueIndex = 0,
  kMultiScaleDeformableAttnInputValueSpatialShapesIndex,
  kMultiScaleDeformableAttnInputValueLevelStartIndex,
  kMultiScaleDeformableAttnInputSamplingLocationsIndex,
  kMultiScaleDeformableAttnInputAttentionWeightsIndex,
  kMultiScaleDeformableAttnInputsNum,
};

enum MultiScaleDeformableAttnOutputIndex : size_t {
  kMultiScaleDeformableAttnOutputAttentionOutIndex = 0,
  kMultiScaleDeformableAttnOutputsNum,
};

abstract::ShapePtr MultiScaleDeformableAttnInferShape(const PrimitivePtr &prim,
                                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto value_shape = input_args[kMultiScaleDeformableAttnInputValueIndex]->GetShape()->GetShapeVector();
  auto sp_loc_shape = input_args[kMultiScaleDeformableAttnInputSamplingLocationsIndex]->GetShape()->GetShapeVector();
  ShapeVector attention_out_shape(3, abstract::Shape::kShapeDimAny);
  attention_out_shape[0] = value_shape[0];
  attention_out_shape[1] = sp_loc_shape[1];
  attention_out_shape[2] = value_shape[1] * value_shape[3];
  return std::make_shared<abstract::Shape>(attention_out_shape);
}

TypePtr MultiScaleDeformableAttnInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto op_name = prim->name();
  std::map<std::string, TypePtr> out_types;
  const std::set<TypePtr> out_valid_types = {kFloat16, kFloat32};
  (void)out_types.emplace("value", input_args[kMultiScaleDeformableAttnInputValueIndex]->BuildType());
  auto type = CheckAndConvertUtils::CheckTensorTypeSame(out_types, out_valid_types, op_name);
  return type;
}

AbstractBasePtr MultiScaleDeformableAttnInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kLessEqual, kMultiScaleDeformableAttnInputsNum, primitive->name());
  auto infer_shape = MultiScaleDeformableAttnInferShape(primitive, input_args);
  auto infer_type = MultiScaleDeformableAttnInferType(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

BaseShapePtr MultiScaleDeformableAttnFunctionV2FuncImpl::InferShape(
  const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  return MultiScaleDeformableAttnInferShape(primitive, input_args);
}

TypePtr MultiScaleDeformableAttnFunctionV2FuncImpl::InferType(const PrimitivePtr &primitive,
                                                              const std::vector<AbstractBasePtr> &input_args) const {
  return MultiScaleDeformableAttnInferType(primitive, input_args);
}

}  // namespace ops
}  // namespace mindspore
