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
#include "ops/matrix_band_part.h"
#include <string>
#include <algorithm>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kXMinShapeSize = 2;

TypePtr MatrixBandPartInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const int64_t kInputNums = 3;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, kInputNums,
                                           prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_type = input_args[kInputIndex0]->BuildType();
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool is_gpu = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice);
  bool is_cpu = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kCPUDevice);
  std::set<TypePtr> valid_types{};
  if (is_gpu || is_cpu) {
    valid_types = common_valid_types_with_complex_and_bool;
  } else {
    valid_types = {kInt32, kInt64, kFloat16, kFloat32, kFloat64};
  }
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTypeValid("lower", input_args[kInputIndex1]->BuildType(), {kInt32, kInt64},
                                             prim_name);
  (void)CheckAndConvertUtils::CheckTypeValid("upper", input_args[kInputIndex2]->BuildType(), {kInt32, kInt64},
                                             prim_name);
  return x_type;
}

abstract::ShapePtr MatrixBandPartInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);

  auto x_shape_ptr = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x_shape_ptr);
  auto lower_shape_ptr = input_args[kInputIndex1]->BuildShape();
  MS_EXCEPTION_IF_NULL(lower_shape_ptr);
  auto upper_shape_ptr = input_args[kInputIndex2]->BuildShape();
  MS_EXCEPTION_IF_NULL(upper_shape_ptr);
  if (x_shape_ptr->IsDynamic() || lower_shape_ptr->IsDynamic() || upper_shape_ptr->IsDynamic()) {
    return x_shape_ptr->cast<abstract::ShapePtr>();
  }

  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("x shape size", SizeToLong(x_shape.size()), kGreaterEqual, kXMinShapeSize,
                                           prim_name);
  auto lower_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto upper_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];

  int64_t batch_rank = 0;
  if (primitive->HasAttr(kBatchRank)) {
    auto value_ptr = primitive->GetAttr(kBatchRank);
    batch_rank = GetValue<int64_t>(value_ptr);
  }
  // Input 'lower' must be a tensor with a value or a scalar.
  (void)CheckAndConvertUtils::CheckInteger("rank of 'lower'", SizeToLong(lower_shape.size()), kEqual, batch_rank,
                                           prim_name);
  // Input 'upper' must be a tensor with a value or a scalar.
  (void)CheckAndConvertUtils::CheckInteger("rank of 'upper'", SizeToLong(upper_shape.size()), kEqual, batch_rank,
                                           prim_name);

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool is_gpu = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice);
  bool is_cpu = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kCPUDevice);
  // Ascend will use the batch_rank to implement vmap feature.
  if (!is_gpu && !is_cpu) {
    return std::make_shared<abstract::Shape>(x_shape);
  }

  auto broadcast_shape = x_shape;
  if (input_args[kInputIndex1]->isa<abstract::AbstractTensor>()) {
    auto expanded_lower_shape = GetExpandedShape<int64_t>(lower_shape, broadcast_shape.size());
    // Check whether broadcasting is possible
    (void)CalBroadCastShape(x_shape, expanded_lower_shape, prim_name, "x", "lower");
    // Get broadcast shape
    broadcast_shape = CalBroadCastShape(broadcast_shape, expanded_lower_shape, prim_name);
  }
  if (input_args[kInputIndex2]->isa<abstract::AbstractTensor>()) {
    auto expanded_upper_shape = GetExpandedShape<int64_t>(upper_shape, broadcast_shape.size());
    // Check whether broadcasting is possible
    (void)CalBroadCastShape(x_shape, expanded_upper_shape, prim_name, "x", "upper");
    // Get broadcast shape
    broadcast_shape = CalBroadCastShape(broadcast_shape, expanded_upper_shape, prim_name);
  }
  return std::make_shared<abstract::Shape>(broadcast_shape);
}
}  // namespace

MIND_API_OPERATOR_IMPL(MatrixBandPart, BaseOperator);
AbstractBasePtr MatrixBandPartInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  auto type = MatrixBandPartInferType(primitive, input_args);
  auto shape = MatrixBandPartInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(MatrixBandPart, prim::kPrimMatrixBandPart, MatrixBandPartInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
