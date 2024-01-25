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

#include "ops/get_squeeze_slice_shape.h"

#include <vector>
#include <string>
#include <algorithm>
#include <memory>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/structure_ops.h"

namespace mindspore {
namespace ops {
namespace {
inline size_t GetOutDim(const PrimitivePtr &primitive, const ShapeVector &data_shape) {
  auto tuple_index_types = GetValue<std::vector<int64_t>>(primitive->GetAttr(kAttrTupleIndexTypes));
  size_t ini_index = 0;
  const size_t max_indices_num = 8;
  for (size_t i = 0; i < max_indices_num; i++) {
    if (tuple_index_types[i] == kObjectTypeTensorType) {
      ini_index += 1;
    }
  }

  return data_shape.size() - ini_index;
}
}  // namespace
MIND_API_OPERATOR_IMPL(GetSqueezeSliceShape, BaseOperator);

AbstractBasePtr GetSqueezeSliceShapeInferInner(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  const AbstractBasePtr &data_abs = input_args[kIndex0];
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(data_abs->GetShape());
  auto data_shape = shape_map[kShape];
  auto any_scalar = std::make_shared<abstract::AbstractScalar>(kValueAny, kInt64);
  if (IsDynamicRank(data_shape)) {
    auto abs_tuple = std::make_shared<abstract::AbstractTuple>(std::vector<abstract::AbstractBasePtr>({any_scalar}));
    return abs_tuple->BroadenToDynamicLenSequence();
  }
  auto new_data_dims = GetOutDim(primitive, data_shape);
  std::vector<abstract::AbstractBasePtr> elements(new_data_dims, any_scalar);
  auto abs_tuple = std::make_shared<abstract::AbstractTuple>(elements);
  return abs_tuple;
}

class MIND_API GetSqueezeSliceShapeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    const AbstractBasePtr &data_abs = input_args[kIndex0];
    auto data_shape = data_abs->GetShape()->GetShapeVector();
    auto new_data_dims = GetOutDim(primitive, data_shape);
    std::vector<abstract::BaseShapePtr> elements(new_data_dims, abstract::kNoShape);
    return std::make_shared<abstract::TupleShape>(elements);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    const AbstractBasePtr &data_abs = input_args[kIndex0];
    auto data_shape = data_abs->GetShape()->GetShapeVector();
    auto new_data_dims = GetOutDim(prim, data_shape);
    std::vector<TypePtr> elements(new_data_dims, kInt64);
    return std::make_shared<Tuple>(elements);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return GetSqueezeSliceShapeInferInner(primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(GetSqueezeSliceShape, prim::kPrimGetSqueezeSliceShape, GetSqueezeSliceShapeInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
