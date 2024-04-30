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

#include "ops/ops_func_impl/moe_finalize_routing.h"
#include <memory>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr size_t kVecSize = 1;
constexpr size_t kMatSize = 2;

constexpr size_t kExpandedX = 0;
constexpr size_t kx1 = 1;
constexpr size_t kx2option = 2;
constexpr size_t kbias = 3;
constexpr size_t kscales = 4;
constexpr size_t krowIdx = 5;
constexpr size_t kexpertIdx = 6;

BaseShapePtr MoeFinalizeRoutingFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kExpandedX]->GetShape());
  auto skip_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kx1]->GetShape());
  auto bias_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kbias]->GetShape());
  auto scales_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kscales]->GetShape());
  auto rowIdx_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[krowIdx]->GetShape());
  auto expertIdx_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kexpertIdx]->GetShape());

  if (x_shape_map.empty() || x_shape_map[kShape].size() != kMatSize) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'X' must be a 2D Tensor type, but got:" << input_args[kExpandedX]->ToString();
  }
  if (skip_shape_map.empty() || skip_shape_map[kShape].size() != kMatSize) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'skip1' must be a 2D Tensor type, but got:" << input_args[kx1]->ToString();
  }
  if (bias_shape_map.empty() || bias_shape_map[kShape].size() != kMatSize) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'bias' must be a 2D Tensor type, but got:" << input_args[kbias]->ToString();
  }
  if (scales_shape_map.empty() || scales_shape_map[kShape].size() != kMatSize) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'scales' must be a 2D Tensor type, but got:" << input_args[kscales]->ToString();
  }
  if (rowIdx_shape_map.empty() || rowIdx_shape_map[kShape].size() != kVecSize) {
    MS_LOG(EXCEPTION) << "For '" << prim_name << "', input 'expanded_row_idx' must be a 1D Tensor type, but got:"
                      << input_args[krowIdx]->ToString();
  }

  if (expertIdx_shape_map.empty() || expertIdx_shape_map[kShape].size() != kMatSize) {
    MS_LOG(EXCEPTION) << "For '" << prim_name << "', input 'expanded_expert_idx' must be a 2D Tensor type, but got:"
                      << input_args[kexpertIdx]->ToString();
  }

  auto x_shp = x_shape_map[kShape];
  auto skip_shp = skip_shape_map[kShape];
  auto bias_shp = bias_shape_map[kShape];
  auto scales_shp = scales_shape_map[kShape];
  auto rowIdx_shp = rowIdx_shape_map[kShape];
  auto expert_shp = expertIdx_shape_map[kShape];

  if (IsDynamicRank(x_shp) || IsDynamicRank(skip_shp) || IsDynamicRank(bias_shp) || IsDynamicRank(scales_shp) ||
      IsDynamicRank(rowIdx_shp) || IsDynamicRank(expert_shp)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  int64_t token_num = expert_shp[0];
  int64_t hidden = x_shp[1];
  std::vector<int64_t> out_shape = {token_num, hidden};
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr MoeFinalizeRoutingFuncImpl::InferType(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  const auto &prim_name = primitive->name();
  const std::set<TypePtr> tensor_valid_types = {kFloat16, kFloat32, kBFloat16};
  const std::set<TypePtr> idx_valid_types = {kInt32};
  const auto &infer_type = input_args[kExpandedX]->GetType();
  MS_EXCEPTION_IF_NULL(infer_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", infer_type, tensor_valid_types, prim_name);

  const auto &skip_type = input_args[kx1]->GetType();
  MS_EXCEPTION_IF_NULL(skip_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("skip1", skip_type, tensor_valid_types, prim_name);

  const auto &bias_type = input_args[kbias]->GetType();
  MS_EXCEPTION_IF_NULL(bias_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("bias", bias_type, tensor_valid_types, prim_name);

  const auto &scales_type = input_args[kscales]->GetType();
  MS_EXCEPTION_IF_NULL(scales_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("scales", scales_type, tensor_valid_types, prim_name);

  const auto &row_idx_type = input_args[krowIdx]->GetType();
  MS_EXCEPTION_IF_NULL(row_idx_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("expanded_row_idx", row_idx_type, idx_valid_types, prim_name);

  const auto &expert_idx_type = input_args[kexpertIdx]->GetType();
  MS_EXCEPTION_IF_NULL(expert_idx_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("expanded_expert_idx", expert_idx_type, idx_valid_types, prim_name);

  return input_args[kExpandedX]->GetType()->Clone();
}

}  // namespace ops
}  // namespace mindspore
