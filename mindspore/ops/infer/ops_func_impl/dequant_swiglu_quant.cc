/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/dequant_swiglu_quant.h"

#include <set>
#include <string>
#include "abstract/ops/primitive_infer_map.h"
#include "primitive/nn_ops.h"
#include "utils/check_convert_utils.h"
#include "ops/primitive_c.h"
#include "mindapi/helper.h"

namespace mindspore {
namespace ops {

bool DequantSwigluQuantFuncImpl::CompareExceptLast(const ShapeVector &a, const ShapeVector &b) const {
  if (a.size() != b.size()) return false;
  if (a.size() < kInputXMinShapeSize) return false;

  // 比较除了最后一个元素以外的所有元素
  for (size_t i = 0; i < a.size() - 1; ++i) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

int32_t DequantSwigluQuantFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                    const InferInfoPtrList &input_infos) const {
  auto &x = input_infos[kDequantSwigluQuantXIndex];
  auto &weight_scale = input_infos[kDequantSwigluQuantWeightScaleIndex];
  auto &activation_scale = input_infos[kDequantSwigluQuantActivationScaleIndex];
  auto &bias = input_infos[kDequantSwigluQuantBiasIndex];
  auto &quant_mode = input_infos[kDequantSwigluQuantQuantModeIndex];
  int64_t last_dim_x = 0;
  if (!x->IsNone() && !x->IsDynamicRank()) {
    MS_CHECK_VALUE(!x->GetShape().empty(), "For DequantSwigluQuant, x shape cannot be null");
    last_dim_x = x->GetShape().back();
    MS_CHECK_VALUE(x->GetShape().size() > kDim1 && (last_dim_x % kDivTwo == 0 || last_dim_x == -1),
                   CheckAndConvertUtils::FormatCommMsg(
                     "For DequantSwigluQuant, the last dimension of x must be a multiple of 2 or equal to -1,",
                     "and the dimension of x must be greater than 1, but got shape ", ShapeVectorToStr(x->GetShape())));
  }

  if (!weight_scale->IsNone() && !weight_scale->IsDynamicRank()) {
    int64_t weight_scale_last_dim = weight_scale->GetShape().back();
    MS_CHECK_VALUE(
      weight_scale->GetShape().size() <= kDim2 && (weight_scale_last_dim == last_dim_x || weight_scale_last_dim == -1),
      CheckAndConvertUtils::FormatCommMsg(
        "For DequantSwigluQuant, the weight_scale dim must less than 2, and last dimension must be the "
        "same as the last dimension of x or -1, but got shape ",
        ShapeVectorToStr(weight_scale->GetShape())));
  }
  if (!activation_scale->IsNone() && !activation_scale->IsDynamicRank()) {
    int64_t activation_scale_last_dim = activation_scale->GetShape().back();
    MS_CHECK_VALUE(CompareExceptLast(activation_scale->GetShape(), x->GetShape()) &&
                     (activation_scale_last_dim == 1 || activation_scale_last_dim == -1),
                   CheckAndConvertUtils::FormatCommMsg(
                     "For DequantSwigluQuant, the last dimension of activation_scale must be 1 or -1, and rest must be",
                     "equal to x, but got shape ", ShapeVectorToStr(activation_scale->GetShape())));
  }
  if (!bias->IsNone() && !bias->IsDynamicRank()) {
    MS_CHECK_VALUE(bias->GetShape().size() == 1 && (bias->GetShape()[0] == last_dim_x || bias->GetShape()[0] == -1),
                   CheckAndConvertUtils::FormatCommMsg(
                     "For DequantSwigluQuant, the bias shape must be (1,) and value must be the same as the last",
                     " dimension of x, or the pointer is null, but got shape ", ShapeVectorToStr(bias->GetShape())));
  }
  auto quant_mode_value = quant_mode->GetScalarValue<std::int64_t>().value();
  MS_CHECK_VALUE(quant_mode_value == kQuantModeStatic || quant_mode_value == kQuantModeDynamic,
                 CheckAndConvertUtils::FormatCommMsg(
                   "For DequantSwigluQuant, quant_mode only support dynamic or static, but got", quant_mode_value));

  return OP_CHECK_SUCCESS;
}

ShapeArray DequantSwigluQuantFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                  const InferInfoPtrList &input_infos) const {
  MS_EXCEPTION_IF_NULL(primitive);
  if (input_infos[kDequantSwigluQuantXIndex]->IsDynamicRank()) {
    ShapeVector y_shape_rank_any = {abstract::TensorShape::kShapeRankAny};
    ShapeVector scale_shape_rank_any = {abstract::TensorShape::kShapeDimAny};
    return {y_shape_rank_any, scale_shape_rank_any};
  }
  int64_t token_num = input_infos[kDequantSwigluQuantXIndex]->GetShape()[0];
  int64_t double_H = input_infos[kDequantSwigluQuantXIndex]->GetShape()[1];
  if (double_H == abstract::Shape::kShapeDimAny) {
    ShapeVector y_shape_dim_any = {token_num, abstract::Shape::kShapeDimAny};
    ShapeVector scale_shape_dim_any = {token_num};
    return {y_shape_dim_any, scale_shape_dim_any};
  }
  ShapeVector y_shape = {token_num, double_H / 2};
  ShapeVector scale_shape = {token_num};
  return {y_shape, scale_shape};
}

std::vector<TypeId> DequantSwigluQuantFuncImpl::InferType(const PrimitivePtr &prim,
                                                          const InferInfoPtrList &input_infos) const {
  auto x_type = input_infos[kDequantSwigluQuantXIndex]->GetType();
  auto weight_scale_type = input_infos[kDequantSwigluQuantWeightScaleIndex]->GetType();
  auto activation_scale_type = input_infos[kDequantSwigluQuantActivationScaleIndex]->GetType();
  if (x_type != kNumberTypeInt32 || weight_scale_type != kNumberTypeFloat32 ||
      activation_scale_type != kNumberTypeFloat32) {
    MS_LOG(EXCEPTION) << "'DequantSwigluQuant' only support [ x int32, weight_scale float32,"
                      << " activation_scale float32 ]. Others are optional, but got x type: " << x_type
                      << " weight_scale type: " << weight_scale_type
                      << "activation_scale type: " << activation_scale_type;
  }
  auto y_out_dtype = kNumberTypeInt8;
  auto scale_out_dtype = kNumberTypeFloat32;
  return {y_out_dtype, scale_out_dtype};
}
}  // namespace ops
}  // namespace mindspore
