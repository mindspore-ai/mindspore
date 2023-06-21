/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "ops/map_tensor_get_values.h"

#include <memory>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ir/anf.h"
#include "ir/dtype/tensor_type.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/sparse_tensor_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(MapTensorGetValues, BaseOperator);

abstract::ShapePtr MapTensorGetValuesInferShape(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) {
  // Check argument abstracts.
  auto abs_map_tensor =
    CheckAndConvertUtils::CheckArgs<abstract::AbstractMapTensor>(kNameMapTensorGetValues, input_args, kInputIndex0);
  auto map_tensor_type = abs_map_tensor->map_tensor_type();
  MS_EXCEPTION_IF_NULL(map_tensor_type);
  auto value_shape_ptr = abs_map_tensor->value_shape();
  MS_EXCEPTION_IF_NULL(value_shape_ptr);
  const auto &value_shape = value_shape_ptr->shape();
  // We don't know the map size in compile time.
  ShapeVector shape_vec = {abstract::Shape::kShapeDimAny};
  (void)shape_vec.insert(shape_vec.end(), value_shape.begin(), value_shape.end());
  return std::make_shared<abstract::Shape>(shape_vec);
}

TypePtr MapTensorGetValuesInferType(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) {
  auto abs_map_tensor =
    CheckAndConvertUtils::CheckArgs<abstract::AbstractMapTensor>(kNameMapTensorGetValues, input_args, kInputIndex0);
  auto map_tensor_type = abs_map_tensor->map_tensor_type();
  MS_EXCEPTION_IF_NULL(map_tensor_type);
  const auto &value_dtype = map_tensor_type->value_dtype();
  return value_dtype;
}

AbstractBasePtr MapTensorGetValuesInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  // Check number of arguments.
  constexpr int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, kNameMapTensorGetValues);
  auto value_dtype = MapTensorGetValuesInferType(primitive, input_args);
  auto shape = MapTensorGetValuesInferShape(primitive, input_args);
  return std::make_shared<abstract::AbstractTensor>(value_dtype, shape);
}

// AG means auto generated
class MIND_API AGMapTensorGetValuesInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MapTensorGetValuesInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MapTensorGetValuesInferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MapTensorGetValuesInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MapTensorGetValues, prim::kPrimMapTensorGetValues, AGMapTensorGetValuesInfer, false);
}  // namespace ops
}  // namespace mindspore
