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
#include "ops/ops_func_impl/multi_scale_deformable_attn_grad.h"
#include "ops/auto_generate/gen_lite_ops.h"

namespace mindspore {
namespace ops {
enum MultiScaleDeformableAttnGradInputIndex : size_t {
  kMultiScaleDeformableAttnGradInputValueIndex = 0,
  kMultiScaleDeformableAttnGradInputSpatialShapesIndex,
  kMultiScaleDeformableAttnGradInputLevelStartIndex,
  kMultiScaleDeformableAttnGradInputSamplingLocIndex,
  kMultiScaleDeformableAttnGradInputAttnWeightIndex,
  kMultiScaleDeformableAttnGradInputGradOutputIndex,
  kMultiScaleDeformableAttnGradInputsNum,
};

enum MultiScaleDeformableAttnGradOutputIndex : size_t {
  kMultiScaleDeformableAttnGradOutputGradValueIndex = 0,
  kMultiScaleDeformableAttnGradOutputGradSamplingLocIndex,
  kMultiScaleDeformableAttnGradOutputGradAttnWeightIndex,
  kMultiScaleDeformableAttnGradOutputsNum,
};

abstract::TupleShapePtr MultiScaleDeformableAttnGradInferShape(const PrimitivePtr &prim,
                                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto value_shape = input_args[kMultiScaleDeformableAttnGradInputValueIndex]->GetShape()->GetShapeVector();
  auto sp_loc_shape = input_args[kMultiScaleDeformableAttnGradInputSamplingLocIndex]->GetShape()->GetShapeVector();

  auto out_one_shape = {value_shape[0], value_shape[1], value_shape[2], value_shape[3]};
  auto out_two_shape = {sp_loc_shape[0], sp_loc_shape[1], sp_loc_shape[2],
                        sp_loc_shape[3], sp_loc_shape[4], sp_loc_shape[5]};
  auto out_three_shape = {sp_loc_shape[0], sp_loc_shape[1], sp_loc_shape[2], sp_loc_shape[3], sp_loc_shape[5]};

  abstract::BaseShapePtrList out_shape = std::vector<abstract::BaseShapePtr>{
    std::make_shared<abstract::Shape>(out_one_shape), std::make_shared<abstract::Shape>(out_two_shape),
    std::make_shared<abstract::Shape>(out_three_shape)};
  return std::make_shared<abstract::TupleShape>(out_shape);
}

TuplePtr MultiScaleDeformableAttnGradInferType(const PrimitivePtr &prim,
                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto op_name = prim->name();
  std::map<std::string, TypePtr> out_types;
  const std::set<TypePtr> out_valid_types = {kFloat16, kFloat32};
  (void)out_types.emplace("value", input_args[kMultiScaleDeformableAttnGradInputValueIndex]->BuildType());
  auto type = CheckAndConvertUtils::CheckTensorTypeSame(out_types, out_valid_types, op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{type, type, type});
}

AbstractBasePtr MultiScaleDeformableAttnGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kLessEqual, kMultiScaleDeformableAttnGradInputsNum,
                                       primitive->name());
  auto infer_shape = MultiScaleDeformableAttnGradInferShape(primitive, input_args);
  auto infer_type = MultiScaleDeformableAttnGradInferType(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

BaseShapePtr MultiScaleDeformableAttentionV2GradFuncImpl::InferShape(
  const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  return MultiScaleDeformableAttnGradInferShape(primitive, input_args);
}

TypePtr MultiScaleDeformableAttentionV2GradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                               const std::vector<AbstractBasePtr> &input_args) const {
  return MultiScaleDeformableAttnGradInferType(primitive, input_args);
}

}  // namespace ops
}  // namespace mindspore
