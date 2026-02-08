/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/batch_norm_gather_stats_with_counts.h"
#include <set>
#include <vector>
#include <memory>
#include <utility>
#include "primitive/op_name.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "infer/ops_func_impl/common_infer_fns.h"

namespace mindspore {
namespace ops {
constexpr size_t kInputShapeMinShapeBNGSWC = 2;
BaseShapePtr BatchNormGatherStatsWithCountsFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                                const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  ShapeVector final_mean_all_shape;
  ShapeVector final_invstd_all_shape;
  if (IsDynamicRank(input_shape)) {
    ShapeVector dyn_mean_all_shape{abstract::TensorShape::kShapeDimAny};
    ShapeVector dyn_invstd_all_shape{abstract::TensorShape::kShapeDimAny};
    final_mean_all_shape = dyn_mean_all_shape;
    final_invstd_all_shape = dyn_invstd_all_shape;
  } else {
    MS_CHECK_VALUE(input_shape.size() >= kInputShapeMinShapeBNGSWC,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("input shape", SizeToLong(input_shape.size()),
                                                               kGreaterEqual, kInputShapeMinShapeBNGSWC, primitive));
    ShapeVector mean_all_shape{input_shape[kIndex1]};
    ShapeVector invstd_all_shape{input_shape[kIndex1]};
    final_mean_all_shape = mean_all_shape;
    final_invstd_all_shape = invstd_all_shape;
  }
  auto final_mean_all_shape_ptr = std::make_shared<abstract::TensorShape>(std::move(final_mean_all_shape));
  auto final_invstd_all_shape_ptr = std::make_shared<abstract::TensorShape>(std::move(final_invstd_all_shape));
  std::vector<abstract::BaseShapePtr> shape_tuple{final_mean_all_shape_ptr, final_invstd_all_shape_ptr};
  return std::make_shared<abstract::TupleShape>(shape_tuple);
}

TypePtr BatchNormGatherStatsWithCountsFuncImpl::InferType(const PrimitivePtr &primitive,
                                                          const std::vector<AbstractBasePtr> &input_args) const {
  auto op_name = primitive->name();
  const std::set<TypePtr> tensor_valid_types = {kFloat32, kFloat16, kBFloat16};
  TypePtr input_type = input_args[kInputIndex0]->GetType();
  (void)CheckAndConvertUtils::CheckTypeValid("input", input_type, tensor_valid_types, op_name);
  TypePtr mean_type = input_args[kInputIndex1]->GetType();
  (void)CheckAndConvertUtils::CheckTypeValid("mean", mean_type, tensor_valid_types, op_name);
  TypePtr invstd_type = input_args[kInputIndex2]->GetType();
  (void)CheckAndConvertUtils::CheckTypeValid("invstd", invstd_type, tensor_valid_types, op_name);
  if (!IsOptionalInputNone(input_args[kInputIndex3])) {
    TypePtr running_mean_type = input_args[kInputIndex3]->GetType();
    (void)CheckAndConvertUtils::CheckTypeValid("running_mean", running_mean_type, tensor_valid_types, op_name);
  }
  if (!IsOptionalInputNone(input_args[kInputIndex4])) {
    TypePtr running_var_type = input_args[kInputIndex4]->GetType();
    (void)CheckAndConvertUtils::CheckTypeValid("running_var", running_var_type, tensor_valid_types, op_name);
  }
  if (!IsOptionalInputNone(input_args[kInputIndex7])) {
    (void)CheckAndConvertUtils::CheckTypeValid("counts", input_args[kInputIndex7]->GetType(), tensor_valid_types,
                                               op_name);
  } else {
    MS_LOG(EXCEPTION) << "For '" << op_name
                      << "', the type of 'counts' must be Tensor[kFloat32, kFloat16, kBFloat16], but got None.";
  }

  TypePtr output_type = PromoteType(PromoteType(input_type, mean_type, op_name), invstd_type, op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{output_type, output_type});
}
}  // namespace ops
}  // namespace mindspore
