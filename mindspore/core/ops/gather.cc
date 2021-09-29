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

#include "ops/gather.h"

#include <set>
#include <memory>
#include <algorithm>

#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
// gather
AbstractBasePtr GatherInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string &op_name = primitive->name();
  constexpr size_t input_num = 3;
  abstract::CheckArgsSize(op_name, input_args, input_num);
  abstract::AbstractTensorPtr params =
    CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 0);
  abstract::AbstractTensorPtr indices =
    CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 1);
  // check
  std::set<TypePtr> valid_params_types = {kTensorType};
  (void)CheckAndConvertUtils::CheckSubClass("params_type", input_args[kInputIndex0]->BuildType(), valid_params_types,
                                            op_name);
  std::set<TypePtr> int_types = {kInt8, kInt16, kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("index_type", input_args[kInputIndex1]->BuildType(), int_types,
                                                   op_name);
  (void)CheckAndConvertUtils::CheckTypeValid("axis_type", input_args[kInputIndex2]->BuildType(), int_types, op_name);

  bool ind_dyn = (!indices->shape()->min_shape().empty() && !indices->shape()->max_shape().empty());
  bool param_dyn = (!params->shape()->min_shape().empty() && !params->shape()->max_shape().empty());
  int64_t axis_val = 0;
  // 3rd input is a Tensor when Gather is a dynamic shape operator
  if (input_args[kInputIndex2]->isa<abstract::AbstractTensor>()) {
    auto axis = input_args[kInputIndex2]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(axis);
    auto axis_value_ptr = axis->BuildValue();
    MS_EXCEPTION_IF_NULL(axis_value_ptr);
    auto axis_tensor = axis_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(axis_tensor);
    axis_val = *static_cast<int64_t *>(axis_tensor->data_c());
  } else if (input_args[kInputIndex2]->isa<abstract::AbstractScalar>()) {
    auto axis = input_args[kInputIndex2]->cast<abstract::AbstractScalarPtr>();
    axis_val = GetValue<int64_t>(axis->BuildValue());
  } else {
    MS_LOG(EXCEPTION) << "Invalid abstract type:" << input_args[kInputIndex2]->type_name();
  }
  auto params_shp = params->shape()->shape();
  auto indices_shp = indices->shape()->shape();
  auto params_rank = static_cast<int64_t>(params_shp.size());
  CheckAndConvertUtils::CheckInRange<int64_t>("axis", axis_val, kIncludeLeft, {-params_rank, params_rank}, op_name);
  // either inputs or both can be dynamic and computation requires min/max shapes for both
  ShapeVector param_shp_min = (param_dyn) ? params->shape()->min_shape() : params->shape()->shape();
  ShapeVector param_shp_max = (param_dyn) ? params->shape()->max_shape() : params->shape()->shape();
  ShapeVector indices_shp_min = (ind_dyn) ? indices->shape()->min_shape() : indices->shape()->shape();
  ShapeVector indices_shp_max = (ind_dyn) ? indices->shape()->max_shape() : indices->shape()->shape();
  // check axis_val within interval: [-params_rank, params_rank)
  if (!(-params_rank <= axis_val) || !(axis_val < params_rank)) {
    MS_LOG(EXCEPTION) << "For Gather - Axis value must be within [ " << -params_rank << ", " << params_rank << " ) "
                      << "Got " << axis_val << ".";
  }
  if (axis_val < 0) {
    axis_val += params_rank;
  }
  auto calc_shape = [axis_val](const ShapeVector &ind_vec, const ShapeVector &params_vec) -> ShapeVector {
    ShapeVector out_vec;
    (void)std::copy(params_vec.begin(), params_vec.begin() + axis_val, std::back_inserter(out_vec));
    (void)copy(ind_vec.begin(), ind_vec.end(), std::back_inserter(out_vec));
    (void)copy(params_vec.begin() + axis_val + 1, params_vec.end(), std::back_inserter(out_vec));
    return out_vec;
  };
  ShapeVector out_shape = calc_shape(indices_shp, params_shp);
  if (ind_dyn || param_dyn) {
    ShapeVector min_shape = calc_shape(indices_shp_min, param_shp_min);
    ShapeVector max_shape = calc_shape(indices_shp_max, param_shp_max);
    return abstract::MakeAbstract(std::make_shared<abstract::Shape>(out_shape, min_shape, max_shape),
                                  params->BuildType());
  }
  return abstract::MakeAbstract(std::make_shared<abstract::Shape>(out_shape), params->BuildType());
}
REGISTER_PRIMITIVE_EVAL_IMPL(Gather, prim::kPrimGather, GatherInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
