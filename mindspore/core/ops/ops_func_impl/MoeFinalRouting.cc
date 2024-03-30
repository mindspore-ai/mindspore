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

#include "ops/ops_func_impl/minimum.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {

constexpr size_t kMatSize = 2;
constexpr size_t kx = 0;
constexpr size_t kSkip = 1;
constexpr size_t koption = 2;
constexpr size_t kbias = 3;
constexpr size_t kscales = 4;
constexpr size_t krowIdx= 5;
constexpr size_t kexpertIdx= 6;

BaseShapePtr MoeFinalRoutingFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);

  auto prim_name = primitive->name();

  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kx]->GetShape());
  auto skip_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kSkip]->GetShape());
  auto option_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[koption]->GetShape());
  auto bias_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kbias]->GetShape());
  auto scales_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kscales]->GetShape());
  auto rowIdx_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[krowIdx]->GetShape());
  auto expertIdx_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kexpertIdx]->GetShape());

  if (x_shape_map.empty()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'X' must be a Tensor type, but got:" << input_args[kx]->ToString();
  }
  if (skip_shape_map.empty()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'resX' must be a Tensor type, but got:" << input_args[kSkip]->ToString();
  }
  if (option_shape_map.empty()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'option' must be a Tensor type, but got:" << input_args[koption]->ToString();
  }
  if (bias_shape_map.empty()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'bias' must be a Tensor type, but got:" << input_args[kbias]->ToString();
  }
  if (scales_shape_map.empty()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'scales' must be a Tensor type, but got:" << input_args[kscales]->ToString();
  }
  if (rowIdx_shape_map.empty()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'rowIdx' must be a Tensor type, but got:" << input_args[krowIdx]->ToString();
  }

  if (expertIdx_shape_map.empty()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'expertIdx' must be a Tensor type, but got:" << input_args[kexpertIdx]->ToString();
  }

  auto x_shp = x_shape_map[kShape];
  auto skip_shp = skip_shape_map[kShape];
  auto option_shp = option_shape_map[kShape];
  auto bias_shp = bias_shape_map[kShape];
  auto scales_shp = scales_shape_map[kShape];
  auto rowIdx_shp = rowIdx_shape_map[kShape];
  auto expert_shp = expertIdx_shape_map[kShape];

  if (IsDynamicRank(x_shp) || IsDynamicRank(skip_shp) || IsDynamicRank(option_shp) || IsDynamicRank(bias_shp) || IsDynamicRank(scales_shp) || IsDynamicRank(rowIdx_shp) || IsDynamicRank(expert_shp)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  return std::make_shared<abstract::Shape>(skip_shp);
}

TypePtr MoeFinalRoutingFuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  const auto &prim_name = primitive->name();
  const auto &infer_type = input_args[kx]->GetType();
  MS_EXCEPTION_IF_NULL(infer_type);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", infer_type, valid_types, prim_name);

  const auto &skip_type = input_args[kSkip]->GetType();
  MS_EXCEPTION_IF_NULL(skip_type);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("skip", skip_type, valid_types, prim_name);

  const auto &option_type = input_args[koption]->GetType();
  MS_EXCEPTION_IF_NULL(option_type);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("option", option_type, valid_types, prim_name);

  const auto &bias_type = input_args[kbias]->GetType();
  MS_EXCEPTION_IF_NULL(bias_type);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("bias", bias_type, valid_types, prim_name);

  const auto &scales_type = input_args[kscales]->GetType();
  MS_EXCEPTION_IF_NULL(scales_type);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("scales", scales_type, valid_types, prim_name);

  const auto &row_idx_type = input_args[krowIdx]->GetType();
  MS_EXCEPTION_IF_NULL(row_idx_type);
  const std::set<TypePtr> valid_types = {kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("row_idx", row_idx_type, valid_types, prim_name);

    const auto &expert_idx_type = input_args[kexpertIdx]->GetType();
  MS_EXCEPTION_IF_NULL(expert_idx_type);
  const std::set<TypePtr> valid_types = {kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("expert_idx", expert_idx_type, valid_types, prim_name);

  return input_args[kSkip]->GetType()->Clone();
}

}  // namespace ops
}  // namespace mindspore
