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

#include "infer/ops_func_impl/fused_add_topk_div.h"
#include <set>
#include <string>
#include <utility>
#include "abstract/ops/primitive_infer_map.h"
#include "primitive/nn_ops.h"
#include "utils/check_convert_utils.h"
#include "mindapi/helper.h"

namespace mindspore {
namespace ops {
BaseShapePtr FusedAddTopKDivFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  auto op_name = primitive->name();
  auto ordinary_input_num = CheckAndConvertUtils::GetRemoveUMonadAbsNum(input_args);
  (void)CheckAndConvertUtils::CheckInteger("inputs num", SizeToLong(ordinary_input_num), kEqual,
                                           kFusedAddTopKDivInputsNum, op_name);
  auto x_shape_ptr = input_args[kFusedAddTopKDivXIndex]->GetShape();
  if (MS_UNLIKELY(IsDynamicRank(x_shape_ptr->GetShapeVector()))) {
    ShapeVector dyn_output{abstract::Shape::kShapeRankAny};
    return std::make_shared<abstract::Shape>(std::move(dyn_output));
  }

  auto add_num_shape_ptr = input_args[kFusedAddTopKDivAddNumIndex]->GetShape();
  if (MS_UNLIKELY(IsDynamicRank(add_num_shape_ptr->GetShapeVector()))) {
    ShapeVector dyn_output{abstract::Shape::kShapeRankAny};
    return std::make_shared<abstract::Shape>(std::move(dyn_output));
  }

  auto a = x_shape_ptr->GetShapeVector()[0];
  auto k = GetScalarValue<int64_t>(input_args[kFusedAddTopKDivKIndex]->GetValue());
  if (MS_UNLIKELY(!k.has_value())) {
    ShapeVector dyn_output{abstract::Shape::kShapeRankAny};
    return std::make_shared<abstract::Shape>(std::move(dyn_output));
  }

  ShapeVector weight_indices_shape{a, k.value()};
  auto output_shape = std::make_shared<abstract::TensorShape>(weight_indices_shape);
  return std::make_shared<abstract::TupleShape>(abstract::BaseShapePtrList({output_shape, output_shape}));
}

TypePtr FusedAddTopKDivFuncImpl::InferType(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto weight_out_type = std::make_shared<TensorType>(kFloat32);
  auto indices_out_type = std::make_shared<TensorType>(kInt32);
  return std::make_shared<Tuple>(std::vector<TypePtr>{weight_out_type, indices_out_type});
}
}  // namespace ops
}  // namespace mindspore
